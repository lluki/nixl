# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process, batch-only NIXLShard data-plane agent.

The first implementation milestone intentionally supports a local logical
device only.  It establishes the fixed-file, registered-buffer, operation
handle, and completion contracts that the remote UCX path will reuse.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import stat
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence

from nixl import _bindings as nixl_bindings
from nixl import _utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
from nixl._bindings import DRAM_SEG, FILE_SEG, nixlXferDList

from .transport import (
    RemoteDevice,
    TcpEndpoint,
    TcpShardServer,
    TcpShardTransport,
    TransportError,
)


class CompletionStatus(str, Enum):
    OK = "OK"
    CANCELLED = "CANCELLED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    IO_ERROR = "IO_ERROR"
    DATA_LOSS = "DATA_LOSS"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True)
class AgentConfig:
    file_path: str | os.PathLike[str]
    size: int
    logical_device_id: int
    logical_device_generation: int = 1
    authority_epoch: int | None = None
    io_terminal_callback: Callable[[dict[str, Any]], None] | None = field(
        default=None, compare=False, repr=False
    )
    create: bool = False
    direct_io: bool = True
    alignment: int = 4096
    max_inflight_ops: int = 128
    listen_port: int = 0
    agent_name: str | None = None
    remote_devices: tuple[RemoteDevice, ...] = ()
    tcp_listen_host: str | None = None
    tcp_listen_port: int = 0
    tcp_request_timeout: float = 5.0
    tcp_max_request_bytes: int = 64 * 1024 * 1024
    tcp_max_staging_bytes: int = 256 * 1024 * 1024
    tcp_max_workers: int = 8


@dataclass(frozen=True)
class BufferRegion:
    address: int
    length: int
    device_id: int = 0
    owner: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class MemoryHandle:
    id: str


@dataclass(frozen=True)
class OperationHandle:
    id: str


class OperationTimeoutError(TimeoutError):
    """A bounded wait expired; handles remain valid for directed recovery."""

    def __init__(self, handles: Sequence[OperationHandle]):
        super().__init__("operations did not become terminal before timeout")
        self.handles = tuple(handles)


@dataclass(frozen=True)
class IoItem:
    device_id: int
    device_offset: int
    memory: MemoryHandle
    memory_index: int
    memory_offset: int
    length: int
    device_generation: int | None = None
    authority_epoch: int | None = None
    terminal_report: dict[str, Any] | None = field(
        default=None, compare=False, repr=False
    )
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Completion:
    handle: OperationHandle
    status: CompletionStatus
    bytes_transferred: int
    detail: str = ""
    terminal: bool = True


@dataclass(frozen=True)
class AgentMetrics:
    submitted_ops: int
    completed_ops: int
    bytes_transferred: int
    error_ops: int
    cancelled_ops: int


@dataclass
class _RegisteredMemory:
    regions: tuple[BufferRegion, ...]
    nixl_descs: Any


@dataclass(frozen=True)
class _PendingItem:
    handle: OperationHandle
    memory_id: str
    length: int
    terminal_report: dict[str, Any] | None = None


@dataclass
class _PendingBatch:
    nixl_handle: Any
    items: tuple[_PendingItem, ...]
    last_progress_error: str = ""
    force_error: bool = False
    terminal_status: CompletionStatus | None = None


@dataclass
class _RemotePendingBatch:
    future: Future[list[tuple[CompletionStatus, int, str]]]
    items: tuple[_PendingItem, ...]


def buffer_region_from_writable(buffer: Any) -> BufferRegion:
    """Return a directly addressable region while retaining its Python owner."""

    view = memoryview(buffer)
    if view.readonly:
        raise ValueError("buffer must be writable")
    if not view.c_contiguous:
        raise ValueError("buffer must be C-contiguous")
    byte_view = view.cast("B")
    address = ctypes.addressof(ctypes.c_char.from_buffer(byte_view))
    return BufferRegion(address=address, length=byte_view.nbytes, owner=byte_view)


class ShardAgent:
    """One in-process agent managing one preallocated logical-device file."""

    def __init__(self, config: AgentConfig):
        self._config = config
        self._validate_config(config)
        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)
        self._state = "OPEN"
        self._memory: dict[str, _RegisteredMemory] = {}
        self._pending: OrderedDict[str, _PendingBatch] = OrderedDict()
        self._remote_pending: OrderedDict[str, _RemotePendingBatch] = OrderedDict()
        self._pending_handles: dict[str, str] = {}
        self._completed: OrderedDict[str, Completion] = OrderedDict()
        self._cancel_requested: set[str] = set()
        self._metrics = {
            "submitted_ops": 0,
            "completed_ops": 0,
            "bytes_transferred": 0,
            "error_ops": 0,
            "cancelled_ops": 0,
        }
        self._remote_devices = {
            remote.logical_device_id: remote for remote in config.remote_devices
        }
        self._fd = self._open_file(config)
        self._remote_executor = ThreadPoolExecutor(
            max_workers=config.tcp_max_workers,
            thread_name_prefix="nixlshard-remote",
        )
        self._transport = TcpShardTransport(
            timeout=config.tcp_request_timeout,
            max_request_bytes=config.tcp_max_request_bytes,
            max_staging_bytes=config.tcp_max_staging_bytes,
        )
        self._tcp_server: TcpShardServer | None = None
        self._safe_posix_error_release = False

        try:
            agent_config = nixl_agent_config(
                True,
                config.listen_port != 0,
                config.listen_port,
                backends=[],
                sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
            )
            agent_name = config.agent_name or (
                f"nixlshard-{config.logical_device_id}-{uuid.uuid4().hex[:12]}"
            )
            self._nixl = nixl_agent(agent_name, agent_config)
            if "POSIX" not in self._nixl.get_plugin_list():
                raise RuntimeError("NIXL POSIX plugin is not available")
            self._nixl.create_backend("POSIX")
            self._safe_posix_error_release = bool(
                getattr(nixl_bindings, "HAVE_SAFE_POSIX_ERROR_RELEASE_CORE", False)
                and self._nixl.get_backend_params("POSIX").get("safe_error_release")
                == "true"
            )
            self._file_registration = self._nixl.register_memory(
                [(0, config.size, self._fd, f"device-{config.logical_device_id}")],
                "FILE",
                backends=["POSIX"],
            )
            if config.tcp_listen_host is not None:
                self._tcp_server = TcpShardServer(
                    TcpEndpoint(config.tcp_listen_host, config.tcp_listen_port),
                    self._handle_remote_request,
                    timeout=config.tcp_request_timeout,
                    max_request_bytes=config.tcp_max_request_bytes,
                    max_staging_bytes=config.tcp_max_staging_bytes,
                    max_workers=config.tcp_max_workers,
                )
        except Exception:
            try:
                if self._tcp_server is not None:
                    self._tcp_server.close()
                self._remote_executor.shutdown(wait=True, cancel_futures=True)
                if config.create:
                    try:
                        Path(config.file_path).unlink()
                    except FileNotFoundError:
                        pass
            finally:
                os.close(self._fd)
            raise

    @classmethod
    def open(cls, config: AgentConfig) -> ShardAgent:
        return cls(config)

    @property
    def logical_device_id(self) -> int:
        return self._config.logical_device_id

    @property
    def logical_device_generation(self) -> int:
        return self._config.logical_device_generation

    @property
    def size(self) -> int:
        return self._config.size

    @property
    def tcp_endpoint(self) -> TcpEndpoint | None:
        return None if self._tcp_server is None else self._tcp_server.endpoint

    @staticmethod
    def _validate_config(config: AgentConfig) -> None:
        if config.size <= 0:
            raise ValueError("size must be positive")
        if config.logical_device_id < 0:
            raise ValueError("logical_device_id must be non-negative")
        if config.logical_device_generation <= 0:
            raise ValueError("logical_device_generation must be positive")
        if config.authority_epoch is not None and config.authority_epoch <= 0:
            raise ValueError("authority_epoch must be positive")
        if config.alignment <= 0 or config.alignment & (config.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if config.direct_io and config.size % config.alignment:
            raise ValueError("direct-I/O file size must be alignment-sized")
        if config.max_inflight_ops <= 0:
            raise ValueError("max_inflight_ops must be positive")
        if config.tcp_request_timeout <= 0:
            raise ValueError("tcp_request_timeout must be positive")
        if config.tcp_max_request_bytes <= 0 or config.tcp_max_staging_bytes <= 0:
            raise ValueError("TCP byte limits must be positive")
        if config.tcp_max_workers <= 0:
            raise ValueError("tcp_max_workers must be positive")
        remote_ids = [remote.logical_device_id for remote in config.remote_devices]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError("remote logical device ids must be unique")
        if config.logical_device_id in remote_ids:
            raise ValueError("local logical device cannot also be remote")
        for remote in config.remote_devices:
            if remote.logical_device_id < 0:
                raise ValueError("remote logical device id must be non-negative")
            if remote.logical_device_generation <= 0:
                raise ValueError("remote logical device generation must be positive")
            if not remote.host or remote.port <= 0 or remote.port > 65535:
                raise ValueError("remote device endpoint is invalid")

    @staticmethod
    def _open_file(config: AgentConfig) -> int:
        path = Path(config.file_path)
        flags = os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if config.direct_io:
            if not hasattr(os, "O_DIRECT"):
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            flags |= os.O_DIRECT
        if config.create:
            flags |= os.O_CREAT | os.O_EXCL

        fd = os.open(path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            mode = os.fstat(fd)
            if not stat.S_ISREG(mode.st_mode):
                raise ValueError(f"storage path is not a regular file: {path}")
            if config.create:
                os.posix_fallocate(fd, 0, config.size)
                os.fsync(fd)
                mode = os.fstat(fd)
            if mode.st_size != config.size:
                raise ValueError(
                    f"storage file size {mode.st_size} does not match configured size "
                    f"{config.size}"
                )
            return fd
        except Exception:
            os.close(fd)
            if config.create:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            raise

    def register_memory(self, regions: Sequence[BufferRegion]) -> MemoryHandle:
        with self._lock:
            self._ensure_open()
            if not regions:
                raise ValueError("at least one region is required")
            frozen_regions = tuple(regions)
            for region in frozen_regions:
                if region.address <= 0 or region.length <= 0:
                    raise ValueError(
                        "memory regions need a positive address and length"
                    )

            token = uuid.uuid4().hex
            descs = [
                (region.address, region.length, region.device_id, token)
                for region in frozen_regions
            ]
            nixl_descs = self._nixl.register_memory(descs, "DRAM", backends=["POSIX"])
            self._memory[token] = _RegisteredMemory(frozen_regions, nixl_descs)
            return MemoryHandle(token)

    def unregister_memory(self, handle: MemoryHandle) -> None:
        with self._lock:
            self._ensure_open()
            if any(
                item.memory_id == handle.id
                for batch in (*self._pending.values(), *self._remote_pending.values())
                for item in batch.items
            ):
                raise RuntimeError("memory still belongs to an active operation")
            registered = self._memory.get(handle.id)
            if registered is None:
                raise KeyError(f"unknown memory handle: {handle.id}")
            self._nixl.deregister_memory(registered.nixl_descs, backends=["POSIX"])
            del self._memory[handle.id]

    def submit_batch_load(self, items: Sequence[IoItem]) -> list[OperationHandle]:
        return self._submit(items, "READ")

    def submit_batch_store(self, items: Sequence[IoItem]) -> list[OperationHandle]:
        return self._submit(items, "WRITE")

    def load_batch(
        self, items: Sequence[IoItem], timeout: float | None = None
    ) -> list[Completion]:
        return self.wait(self.submit_batch_load(items), timeout)

    def store_batch(
        self, items: Sequence[IoItem], timeout: float | None = None
    ) -> list[Completion]:
        return self.wait(self.submit_batch_store(items), timeout)

    def _submit(self, items: Sequence[IoItem], operation: str) -> list[OperationHandle]:
        with self._lock:
            self._ensure_open()
            if not items:
                raise ValueError("at least one I/O item is required")

            self._progress()
            handles = [OperationHandle(uuid.uuid4().hex) for _ in items]
            self._metrics["submitted_ops"] += len(handles)
            accepted: list[
                tuple[OperationHandle, IoItem, _RegisteredMemory, BufferRegion]
            ] = []
            available = self._config.max_inflight_ops - len(self._pending_handles)

            for public_handle, item in zip(handles, items, strict=True):
                status, detail, registered = self._validate_item(item)
                if status is not None:
                    self._complete_immediately(public_handle, status, detail)
                    continue
                assert registered is not None
                region = registered.regions[item.memory_index]
                if len(accepted) >= available:
                    self._complete_immediately(
                        public_handle,
                        CompletionStatus.RESOURCE_EXHAUSTED,
                        "maximum in-flight operation count reached",
                    )
                    continue
                accepted.append((public_handle, item, registered, region))

            if not accepted:
                return handles

            local = [
                entry
                for entry in accepted
                if entry[1].device_id == self._config.logical_device_id
            ]
            remote: dict[
                int,
                list[tuple[OperationHandle, IoItem, _RegisteredMemory, BufferRegion]],
            ] = defaultdict(list)
            for entry in accepted:
                if entry[1].device_id != self._config.logical_device_id:
                    remote[entry[1].device_id].append(entry)

            if local:
                self._start_local_batch(local, operation)
            for device_id, remote_items in remote.items():
                self._start_remote_batch(
                    self._remote_devices[device_id].endpoint,
                    remote_items,
                    operation,
                )
            self._progress()
            return handles

    def _start_local_batch(
        self,
        accepted: Sequence[
            tuple[OperationHandle, IoItem, _RegisteredMemory, BufferRegion]
        ],
        operation: str,
    ) -> None:
        memory_descs = nixlXferDList(
            DRAM_SEG,
            [
                (
                    region.address + item.memory_offset,
                    item.length,
                    region.device_id,
                )
                for _, item, _, region in accepted
            ],
        )
        file_descs = nixlXferDList(
            FILE_SEG,
            [(item.device_offset, item.length, self._fd) for _, item, _, _ in accepted],
        )

        try:
            nixl_handle = self._nixl.initialize_xfer(
                operation,
                memory_descs,
                file_descs,
                self._nixl.name,
                backends=["POSIX"],
            )
        except Exception as exc:
            for public_handle, _, _, _ in accepted:
                self._complete_immediately(
                    public_handle, CompletionStatus.IO_ERROR, str(exc)
                )
            return

        batch_id = uuid.uuid4().hex
        pending_items = tuple(
            _PendingItem(
                public_handle, item.memory.id, item.length, item.terminal_report
            )
            for public_handle, item, _, _ in accepted
        )
        self._pending[batch_id] = _PendingBatch(nixl_handle, pending_items)
        for pending_item in pending_items:
            self._pending_handles[pending_item.handle.id] = batch_id
        try:
            state = self._nixl.transfer(nixl_handle)
        except Exception as exc:
            pending = self._pending[batch_id]
            pending.last_progress_error = str(exc)
            pending.force_error = True
            if self._safe_posix_error_release:
                pending.terminal_status = CompletionStatus.IO_ERROR
        else:
            if state == "DONE":
                self._pending[batch_id].terminal_status = CompletionStatus.OK
            elif state != "PROC":
                pending = self._pending[batch_id]
                pending.last_progress_error = f"NIXL post returned {state}"
                pending.force_error = True
                if self._safe_posix_error_release:
                    pending.terminal_status = CompletionStatus.IO_ERROR

    def _start_remote_batch(
        self,
        endpoint: TcpEndpoint,
        accepted: Sequence[
            tuple[OperationHandle, IoItem, _RegisteredMemory, BufferRegion]
        ],
        operation: str,
    ) -> None:
        batch_id = uuid.uuid4().hex
        pending_items = tuple(
            _PendingItem(
                public_handle, item.memory.id, item.length, item.terminal_report
            )
            for public_handle, item, _, _ in accepted
        )
        future = self._remote_executor.submit(
            self._run_remote_batch, endpoint, tuple(accepted), operation
        )
        self._remote_pending[batch_id] = _RemotePendingBatch(future, pending_items)
        for pending_item in pending_items:
            self._pending_handles[pending_item.handle.id] = batch_id

    def _run_remote_batch(
        self,
        endpoint: TcpEndpoint,
        accepted: tuple[
            tuple[OperationHandle, IoItem, _RegisteredMemory, BufferRegion], ...
        ],
        operation: str,
    ) -> list[tuple[CompletionStatus, int, str]]:
        wire_items = [
            {
                "device_id": item.device_id,
                "device_generation": (
                    item.device_generation
                    if item.device_generation is not None
                    else self._remote_devices[item.device_id].logical_device_generation
                ),
                "authority_epoch": item.authority_epoch,
                "terminal_report": item.terminal_report,
                "device_offset": item.device_offset,
                "length": item.length,
                "request_id": item.request_id,
            }
            for _, item, _, _ in accepted
        ]
        try:

            def store_payload() -> bytes:
                return b"".join(
                    ctypes.string_at(region.address + item.memory_offset, item.length)
                    for _, item, _, region in accepted
                )

            results, response_payload = self._transport.request(
                endpoint,
                "store" if operation == "WRITE" else "load",
                wire_items,
                store_payload if operation == "WRITE" else b"",
            )
            cursor = 0
            converted: list[tuple[CompletionStatus, int, str]] = []
            for result, (_, item, _, region) in zip(results, accepted, strict=True):
                try:
                    status = CompletionStatus(str(result["status"]))
                    transferred = int(result.get("bytes_transferred", 0))
                    detail = str(result.get("detail", ""))
                except (KeyError, TypeError, ValueError) as exc:
                    raise TransportError(
                        "DATA_LOSS", f"invalid remote result: {exc}"
                    ) from exc
                if status is CompletionStatus.OK and transferred != item.length:
                    raise TransportError(
                        "DATA_LOSS",
                        f"successful {operation.lower()} has a short byte count",
                    )
                if operation == "READ" and status is CompletionStatus.OK:
                    ctypes.memmove(
                        region.address + item.memory_offset,
                        response_payload[cursor : cursor + item.length],
                        item.length,
                    )
                cursor += item.length
                converted.append((status, transferred, detail))
            return converted
        except TransportError as exc:
            try:
                status = CompletionStatus(exc.status)
            except ValueError:
                status = CompletionStatus.INTERNAL
            return [(status, 0, exc.detail) for _ in accepted]
        except Exception as exc:
            return [(CompletionStatus.INTERNAL, 0, str(exc)) for _ in accepted]

    def _handle_remote_request(
        self,
        operation: str,
        items: Sequence[dict[str, Any]],
        payload: bytes,
    ) -> tuple[list[dict[str, Any]], bytes]:
        """Execute a TCP fallback request against the owned fixed file.

        TCP is a functional stand-in for the planned UCX staging path.  The
        transport owns the bounded staging buffers; this lock keeps shutdown
        from deregistering and closing the file while a request is active.
        """

        results: list[dict[str, Any]] = []
        response = bytearray()
        payload_cursor = 0
        terminal_reports: dict[tuple[str, ...], dict[str, Any]] = {}
        with self._lock:
            if self._state != "OPEN":
                raise TransportError("UNAVAILABLE", "agent is shutting down")
            for item in items:
                device_id = int(item["device_id"])
                device_generation = int(item["device_generation"])
                authority_epoch = item.get("authority_epoch")
                if authority_epoch is not None:
                    authority_epoch = int(authority_epoch)
                device_offset = int(item["device_offset"])
                length = int(item["length"])
                item_payload = payload[payload_cursor : payload_cursor + length]
                payload_cursor += length
                status = CompletionStatus.OK
                detail = ""
                transferred = 0
                loaded = bytes(length)
                if device_id != self._config.logical_device_id:
                    status = CompletionStatus.UNAVAILABLE
                    detail = "logical device is not hosted by peer"
                elif device_generation != self._config.logical_device_generation:
                    status = CompletionStatus.UNAVAILABLE
                    detail = "logical device generation is stale"
                elif (
                    self._config.authority_epoch is not None
                    and authority_epoch != self._config.authority_epoch
                ):
                    status = CompletionStatus.UNAVAILABLE
                    detail = "naming authority epoch is stale"
                elif device_offset < 0 or device_offset + length > self._config.size:
                    status = CompletionStatus.OUT_OF_RANGE
                    detail = "file range exceeds logical device"
                elif self._config.direct_io and (
                    device_offset % self._config.alignment
                    or length % self._config.alignment
                ):
                    status = CompletionStatus.INVALID_ARGUMENT
                    detail = "unaligned direct I/O"
                else:
                    try:
                        if operation == "store":
                            transferred = self._remote_pwrite(
                                item_payload, device_offset
                            )
                        else:
                            loaded = self._remote_pread(length, device_offset)
                            transferred = len(loaded)
                    except OSError as exc:
                        status = CompletionStatus.IO_ERROR
                        detail = str(exc)
                        transferred = 0
                results.append(
                    {
                        "status": status.value,
                        "bytes_transferred": transferred,
                        "detail": detail,
                    }
                )
                report = item.get("terminal_report")
                if self._config.io_terminal_callback is not None and isinstance(
                    report, dict
                ):
                    # One metadata capability can span multiple scatter/gather
                    # items. Ignore the per-fragment request ID and retain one
                    # report until every item in this request is terminal.
                    report_key = tuple(
                        repr(report.get(field))
                        for field in (
                            "token_kind",
                            "token_id",
                            "service_epoch",
                            "device_id",
                            "agent_epoch",
                            "extent_generation",
                            "device_generation",
                            "client_id",
                        )
                    )
                    terminal_reports.setdefault(report_key, report)
                if operation == "load":
                    response.extend(loaded)
            if self._config.io_terminal_callback is not None:
                for report in terminal_reports.values():
                    try:
                        self._config.io_terminal_callback(report)
                    except Exception:
                        # The initiating client also reports completion. If it
                        # dies, a management fence remains the conservative
                        # fallback for a failed callback.
                        pass
        return results, bytes(response)

    def _remote_pwrite(self, data: bytes, offset: int) -> int:
        if not self._config.direct_io:
            written = 0
            while written < len(data):
                count = os.pwrite(self._fd, data[written:], offset + written)
                if count <= 0:
                    raise OSError("short remote write")
                written += count
            return written

        allocation = nixl_utils.malloc_passthru(len(data))
        try:
            ctypes.memmove(allocation, data, len(data))
            aligned = (ctypes.c_ubyte * len(data)).from_address(allocation)
            count = os.pwritev(self._fd, [memoryview(aligned).cast("B")], offset)
            if count != len(data):
                raise OSError("short remote direct-I/O write")
            return count
        finally:
            nixl_utils.free_passthru(allocation)

    def _remote_pread(self, length: int, offset: int) -> bytes:
        if not self._config.direct_io:
            chunks: list[bytes] = []
            read = 0
            while read < length:
                chunk = os.pread(self._fd, length - read, offset + read)
                if not chunk:
                    raise OSError("short remote read")
                chunks.append(chunk)
                read += len(chunk)
            return b"".join(chunks)

        allocation = nixl_utils.malloc_passthru(length)
        try:
            aligned = (ctypes.c_ubyte * length).from_address(allocation)
            count = os.preadv(self._fd, [memoryview(aligned).cast("B")], offset)
            if count != length:
                raise OSError("short remote direct-I/O read")
            return ctypes.string_at(allocation, length)
        finally:
            nixl_utils.free_passthru(allocation)

    def _validate_item(
        self, item: IoItem
    ) -> tuple[CompletionStatus | None, str, _RegisteredMemory | None]:
        is_local = item.device_id == self._config.logical_device_id
        if not is_local and item.device_id not in self._remote_devices:
            return CompletionStatus.UNAVAILABLE, "logical device is not routable", None
        expected_generation = (
            self._config.logical_device_generation
            if is_local
            else self._remote_devices[item.device_id].logical_device_generation
        )
        if (
            item.device_generation is not None
            and item.device_generation != expected_generation
        ):
            return (
                CompletionStatus.UNAVAILABLE,
                "logical device generation is stale",
                None,
            )
        if (
            self._config.authority_epoch is not None
            and item.authority_epoch != self._config.authority_epoch
        ):
            return (
                CompletionStatus.UNAVAILABLE,
                "naming authority epoch is stale",
                None,
            )
        if item.length <= 0 or item.device_offset < 0 or item.memory_offset < 0:
            return (
                CompletionStatus.INVALID_ARGUMENT,
                "negative offset or invalid length",
                None,
            )
        if is_local and item.device_offset + item.length > self._config.size:
            return (
                CompletionStatus.OUT_OF_RANGE,
                "file range exceeds logical device",
                None,
            )
        registered = self._memory.get(item.memory.id)
        if registered is None:
            return CompletionStatus.INVALID_ARGUMENT, "unknown memory handle", None
        if item.memory_index < 0 or item.memory_index >= len(registered.regions):
            return (
                CompletionStatus.INVALID_ARGUMENT,
                "memory index is out of range",
                None,
            )
        region = registered.regions[item.memory_index]
        if item.memory_offset + item.length > region.length:
            return (
                CompletionStatus.OUT_OF_RANGE,
                "memory range exceeds registration",
                None,
            )
        if is_local and self._config.direct_io:
            address = region.address + item.memory_offset
            values = (item.device_offset, item.length, address)
            if any(value % self._config.alignment for value in values):
                return CompletionStatus.INVALID_ARGUMENT, "unaligned direct I/O", None
        return None, "", registered

    def _complete_immediately(
        self, handle: OperationHandle, status: CompletionStatus, detail: str
    ) -> None:
        self._record_completion(handle, status, 0, detail)

    def _record_completion(
        self,
        handle: OperationHandle,
        status: CompletionStatus,
        bytes_transferred: int,
        detail: str,
        terminal_report: dict[str, Any] | None = None,
    ) -> None:
        if handle.id in self._cancel_requested:
            self._cancel_requested.remove(handle.id)
            status = CompletionStatus.CANCELLED
            bytes_transferred = 0
            detail = "cancellation requested; underlying operation is terminal"
        completion = Completion(handle, status, bytes_transferred, detail)
        self._completed[handle.id] = completion
        self._metrics["completed_ops"] += 1
        self._metrics["bytes_transferred"] += bytes_transferred
        if status is CompletionStatus.CANCELLED:
            self._metrics["cancelled_ops"] += 1
        elif status is not CompletionStatus.OK:
            self._metrics["error_ops"] += 1
        if (
            self._config.io_terminal_callback is not None
            and terminal_report is not None
        ):
            try:
                self._config.io_terminal_callback(terminal_report)
            except Exception:
                pass

    def _progress(self) -> None:
        for batch_id, pending in list(self._pending.items()):
            if pending.terminal_status is None:
                try:
                    state = self._nixl.check_xfer_state(pending.nixl_handle)
                except Exception as exc:
                    pending.last_progress_error = str(exc)
                    pending.force_error = True
                    if self._safe_posix_error_release:
                        pending.terminal_status = CompletionStatus.IO_ERROR
                else:
                    if state == "PROC":
                        continue
                    if state == "DONE":
                        pending.terminal_status = (
                            CompletionStatus.IO_ERROR
                            if pending.force_error
                            else CompletionStatus.OK
                        )
                    else:
                        pending.last_progress_error = f"NIXL status returned {state}"
                        pending.force_error = True
                        if self._safe_posix_error_release:
                            pending.terminal_status = CompletionStatus.IO_ERROR
                        else:
                            continue

            if pending.terminal_status is None:
                continue

            try:
                self._nixl.release_xfer_handle(pending.nixl_handle)
            except Exception as exc:
                pending.last_progress_error = str(exc)
                continue

            self._pending.pop(batch_id, None)
            status = pending.terminal_status
            assert status is not None
            detail = (
                pending.last_progress_error
                if status is CompletionStatus.IO_ERROR
                else ""
            )
            for item in pending.items:
                self._pending_handles.pop(item.handle.id, None)
                transferred = item.length if status is CompletionStatus.OK else 0
                self._record_completion(
                    item.handle, status, transferred, detail, item.terminal_report
                )

        for batch_id, pending in list(self._remote_pending.items()):
            if not pending.future.done():
                continue
            try:
                results = pending.future.result()
            except Exception as exc:
                results = [
                    (CompletionStatus.INTERNAL, 0, str(exc)) for _ in pending.items
                ]
            self._remote_pending.pop(batch_id, None)
            if len(results) != len(pending.items):
                results = [
                    (
                        CompletionStatus.DATA_LOSS,
                        0,
                        "remote result count does not match pending batch",
                    )
                    for _ in pending.items
                ]
            for item, (status, transferred, detail) in zip(
                pending.items, results, strict=True
            ):
                self._pending_handles.pop(item.handle.id, None)
                self._record_completion(
                    item.handle, status, transferred, detail, item.terminal_report
                )

    def cancel(self, handles: Sequence[OperationHandle]) -> None:
        """Request cancellation without releasing buffers before terminal I/O."""

        with self._lock:
            self._ensure_open()
            self._progress()
            unknown = [
                handle.id
                for handle in handles
                if handle.id not in self._pending_handles
                and handle.id not in self._completed
            ]
            if unknown:
                raise KeyError(
                    f"unknown or already consumed operation handles: {unknown}"
                )
            self._cancel_requested.update(
                handle.id for handle in handles if handle.id in self._pending_handles
            )

    def get_metrics(self) -> AgentMetrics:
        with self._lock:
            return AgentMetrics(**self._metrics)

    def poll(self, max_completions: int) -> list[Completion]:
        if max_completions < 0:
            raise ValueError("max_completions must be non-negative")
        with self._lock:
            self._ensure_open()
            self._progress()
            result: list[Completion] = []
            while self._completed and len(result) < max_completions:
                _, completion = self._completed.popitem(last=False)
                result.append(completion)
            return result

    def wait(
        self,
        handles: Sequence[OperationHandle],
        timeout: float | None = None,
    ) -> list[Completion]:
        wanted = [handle.id for handle in handles]
        if len(wanted) != len(set(wanted)):
            raise ValueError("duplicate operation handle")
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self._lock:
                self._ensure_open()
                self._progress()
                if all(operation_id in self._completed for operation_id in wanted):
                    return [
                        self._completed.pop(operation_id) for operation_id in wanted
                    ]
                unknown = [
                    operation_id
                    for operation_id in wanted
                    if operation_id not in self._pending_handles
                    and operation_id not in self._completed
                ]
                if unknown:
                    raise KeyError(
                        f"unknown or already consumed operation handles: {unknown}"
                    )
            if deadline is not None and time.monotonic() >= deadline:
                raise OperationTimeoutError(handles)
            time.sleep(0.0001)

    def close(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._state_changed:
            while self._state == "CLOSING":
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("another thread is still closing the agent")
                self._state_changed.wait(remaining)
            if self._state == "CLOSED":
                return
            self._state = "CLOSING"

        teardown_started = False
        try:
            if self._tcp_server is not None:
                teardown_started = True
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                self._tcp_server.close(timeout=remaining)
            while True:
                with self._lock:
                    self._progress()
                    if not self._pending and not self._remote_pending:
                        teardown_started = True
                        self._teardown_locked()
                        self._state = "CLOSED"
                        self._state_changed.notify_all()
                        return
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("agent still has active operations")
                time.sleep(0.0001)
        except BaseException:
            with self._state_changed:
                if self._state == "CLOSING":
                    self._state = "CLOSE_FAILED" if teardown_started else "OPEN"
                    self._state_changed.notify_all()
            raise

    def _teardown_locked(self) -> None:
        for memory_id, registered in list(self._memory.items()):
            self._nixl.deregister_memory(registered.nixl_descs, backends=["POSIX"])
            del self._memory[memory_id]
        self._nixl.deregister_memory(self._file_registration, backends=["POSIX"])
        os.close(self._fd)
        self._remote_executor.shutdown(wait=True, cancel_futures=False)

    def _ensure_open(self) -> None:
        if self._state != "OPEN":
            raise RuntimeError(f"agent is {self._state.lower()}")

    def __enter__(self) -> ShardAgent:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
