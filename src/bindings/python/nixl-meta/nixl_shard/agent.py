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
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from nixl import _bindings as nixl_bindings
from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
from nixl._bindings import DRAM_SEG, FILE_SEG, nixlXferDList


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
    create: bool = False
    direct_io: bool = True
    alignment: int = 4096
    max_inflight_ops: int = 128
    listen_port: int = 0
    agent_name: str | None = None


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
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class Completion:
    handle: OperationHandle
    status: CompletionStatus
    bytes_transferred: int
    detail: str = ""
    terminal: bool = True


@dataclass
class _RegisteredMemory:
    regions: tuple[BufferRegion, ...]
    nixl_descs: Any


@dataclass(frozen=True)
class _PendingItem:
    handle: OperationHandle
    memory_id: str
    length: int


@dataclass
class _PendingBatch:
    nixl_handle: Any
    items: tuple[_PendingItem, ...]
    last_progress_error: str = ""
    force_error: bool = False
    terminal_status: CompletionStatus | None = None


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
        self._pending_handles: dict[str, str] = {}
        self._completed: OrderedDict[str, Completion] = OrderedDict()
        self._safe_posix_error_release = False
        self._fd = self._open_file(config)

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
        except Exception:
            try:
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
    def size(self) -> int:
        return self._config.size

    @staticmethod
    def _validate_config(config: AgentConfig) -> None:
        if config.size <= 0:
            raise ValueError("size must be positive")
        if config.logical_device_id < 0:
            raise ValueError("logical_device_id must be non-negative")
        if config.alignment <= 0 or config.alignment & (config.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if config.direct_io and config.size % config.alignment:
            raise ValueError("direct-I/O file size must be alignment-sized")
        if config.max_inflight_ops <= 0:
            raise ValueError("max_inflight_ops must be positive")

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
                for batch in self._pending.values()
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
                [
                    (item.device_offset, item.length, self._fd)
                    for _, item, _, _ in accepted
                ],
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
                return handles

            batch_id = uuid.uuid4().hex
            pending_items = tuple(
                _PendingItem(public_handle, item.memory.id, item.length)
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
            self._progress()
            return handles

    def _validate_item(
        self, item: IoItem
    ) -> tuple[CompletionStatus | None, str, _RegisteredMemory | None]:
        if item.device_id != self._config.logical_device_id:
            return CompletionStatus.UNAVAILABLE, "logical device is not local", None
        if item.length <= 0 or item.device_offset < 0 or item.memory_offset < 0:
            return (
                CompletionStatus.INVALID_ARGUMENT,
                "negative offset or invalid length",
                None,
            )
        if item.device_offset + item.length > self._config.size:
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
        if self._config.direct_io:
            address = region.address + item.memory_offset
            values = (item.device_offset, item.length, address)
            if any(value % self._config.alignment for value in values):
                return CompletionStatus.INVALID_ARGUMENT, "unaligned direct I/O", None
        return None, "", registered

    def _complete_immediately(
        self, handle: OperationHandle, status: CompletionStatus, detail: str
    ) -> None:
        self._completed[handle.id] = Completion(handle, status, 0, detail)

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
                self._completed[item.handle.id] = Completion(
                    item.handle, status, transferred, detail
                )

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
            while True:
                with self._lock:
                    self._progress()
                    if not self._pending:
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

    def _ensure_open(self) -> None:
        if self._state != "OPEN":
            raise RuntimeError(f"agent is {self._state.lower()}")

    def __enter__(self) -> ShardAgent:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
