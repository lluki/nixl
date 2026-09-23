# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL UCX payload transport with bounded registered staging allocations."""

from __future__ import annotations

import base64
import ctypes
import socket
import threading
import time
from typing import Any, Callable, Sequence

from nixl import _utils as nixl_utils

from .transport import (
    TcpEndpoint,
    TcpShardServer,
    TransportError,
    _PROTOCOL_VERSION,
    _recv_header,
    _send_frame,
)


class UcxOwnershipError(TransportError):
    """NIXL has not proved that a staging allocation can be released."""


# An ambiguous RDMA operation has no safe reclamation point within this
# process. Keep its allocation and NIXL agent alive until process exit.
_PROCESS_ORPHANS: list[_Staging] = []


class _Staging:
    def __init__(self, agent: Any, length: int):
        self.agent = agent
        self.length = length
        self.address = nixl_utils.malloc_passthru(length)
        try:
            self.registration = agent.register_memory(
                [(self.address, length, 0, "")], "DRAM", backends=["UCX"]
            )
        except BaseException:
            nixl_utils.free_passthru(self.address)
            raise

    def write(self, data: bytes) -> None:
        if len(data) > self.length:
            raise TransportError("INTERNAL", "UCX staging length mismatch")
        ctypes.memmove(self.address, data, len(data))

    def read(self, length: int) -> bytes:
        if length > self.length:
            raise TransportError("INTERNAL", "UCX staging length mismatch")
        return ctypes.string_at(self.address, length)

    def close(self) -> None:
        # A failed deregistration is ambiguous; retain the allocation rather
        # than return an RDMA-visible address to the allocator.
        self.agent.deregister_memory(self.registration, backends=["UCX"])
        nixl_utils.free_passthru(self.address)


class _StagingPool:
    """Reuse published UCX registrations within a fixed byte ceiling."""

    def __init__(self, agent: Any, capacity: int):
        self._agent = agent
        self._capacity = capacity
        self._allocated = 0
        self._free: list[_Staging] = []
        self._orphaned: list[_Staging] = []
        self._condition = threading.Condition()

    def acquire(self, length: int, timeout: float) -> _Staging:
        if length > self._capacity:
            raise TransportError("RESOURCE_EXHAUSTED", "UCX staging budget exhausted")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                fitting = [s for s in self._free if s.length >= length]
                if fitting:
                    stage = min(fitting, key=lambda s: s.length)
                    self._free.remove(stage)
                    return stage
                if self._allocated + length <= self._capacity:
                    self._allocated += length
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError(
                        "RESOURCE_EXHAUSTED", "UCX staging budget exhausted"
                    )
                self._condition.wait(remaining)
        try:
            return _Staging(self._agent, length)
        except BaseException:
            with self._condition:
                self._allocated -= length
                self._condition.notify_all()
            raise

    def release(self, stage: _Staging, *, uncertain: bool = False) -> None:
        with self._condition:
            if uncertain:
                self._orphaned.append(stage)
                _PROCESS_ORPHANS.append(stage)
            else:
                self._free.append(stage)
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            free = self._free
            self._free = []
        first_error: Exception | None = None
        for stage in free:
            try:
                stage.close()
            except Exception as exc:
                self.release(stage, uncertain=True)
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class _UcxPeer:
    def __init__(self, agent: Any):
        self.agent = agent
        self.lock = threading.RLock()
        self.connected: set[str] = set()
        self._metadata_by_peer: dict[str, str] = {}

    def metadata(self) -> str:
        return base64.b64encode(self.agent.get_agent_metadata()).decode("ascii")

    def connect(self, encoded: str, expected_name: str) -> str:
        if self._metadata_by_peer.get(expected_name) == encoded:
            return expected_name
        try:
            # Registrations stay live in the bounded pool, so a changed blob
            # only adds descriptors. NIXL merges that metadata without
            # disconnecting an existing peer or replacing a live descriptor.
            metadata = base64.b64decode(encoded, validate=True)
            peer = self.agent.add_remote_agent(metadata)
            if isinstance(peer, bytes):
                peer = peer.decode()
            if peer != expected_name:
                raise ValueError("UCX peer name does not match control frame")
            if peer not in self.connected:
                self.agent.make_connection(peer, backends=["UCX"])
                self.connected.add(peer)
            self._metadata_by_peer[peer] = encoded
            return peer
        except Exception as exc:
            raise TransportError(
                "UNAVAILABLE", f"UCX connection failed: {exc}"
            ) from exc

    def transfer(
        self, operation: str, local_addr: int, remote_addr: int, length: int, peer: str
    ) -> None:
        local = self.agent.get_xfer_descs([(local_addr, length, 0)], mem_type="DRAM")
        remote = self.agent.get_xfer_descs([(remote_addr, length, 0)], mem_type="DRAM")
        handle = self.agent.initialize_xfer(
            operation, local, remote, peer, backends=["UCX"]
        )
        # The allocation stays registered until NIXL reports terminal state.
        try:
            state = self.agent.transfer(handle)
            while state == "PROC":
                time.sleep(0.0001)
                state = self.agent.check_xfer_state(handle)
        except Exception as exc:
            raise UcxOwnershipError(
                "IO_ERROR", f"UCX transfer terminal state is unknown: {exc}"
            ) from exc
        if state != "DONE":
            raise UcxOwnershipError("IO_ERROR", f"UCX transfer ended in {state}")
        try:
            self.agent.release_xfer_handle(handle)
        except Exception as exc:
            raise UcxOwnershipError(
                "IO_ERROR", f"UCX transfer handle could not be released: {exc}"
            ) from exc


class UcxShardTransport:
    """UCX WRITE/READ for bytes; TCP frames carry only control and results."""

    def __init__(
        self,
        agent: Any,
        *,
        timeout: float,
        max_request_bytes: int,
        max_staging_bytes: int,
    ):
        self._peer = _UcxPeer(agent)
        self._timeout = timeout
        self._max_request_bytes = max_request_bytes
        self._pool = _StagingPool(agent, max_staging_bytes)
        self._transfer_bytes = 0

    @property
    def transfer_bytes(self) -> int:
        with self._peer.lock:
            return self._transfer_bytes

    def request(
        self,
        endpoint: TcpEndpoint,
        operation: str,
        items: Sequence[dict[str, Any]],
        payload: bytes | Callable[[], bytes],
    ) -> tuple[list[dict[str, Any]], bytes]:
        length = sum(int(item["length"]) for item in items)
        if length > self._max_request_bytes:
            raise TransportError("RESOURCE_EXHAUSTED", "UCX staging budget exhausted")
        staging = None
        retain_staging = False
        try:
            staging = self._pool.acquire(length, self._timeout)
            if operation == "store":
                staging.write(payload() if callable(payload) else payload)
            try:
                with socket.create_connection(
                    (endpoint.host, endpoint.port), timeout=self._timeout
                ) as connection:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # Once the request can reach the server, wait for its
                    # terminal reply; the fixed-file I/O may have committed.
                    connection.settimeout(None)
                    _send_frame(
                        connection,
                        {
                            "version": _PROTOCOL_VERSION,
                            "operation": operation,
                            "items": list(items),
                        },
                        b"",
                    )
                    ready, ready_length = _recv_header(connection)
                    if ready_length or ready.get("version") != _PROTOCOL_VERSION:
                        raise TransportError("DATA_LOSS", "invalid UCX ready frame")
                    if ready.get("error_status"):
                        raise TransportError(
                            str(ready["error_status"]),
                            str(ready.get("detail", "")),
                        )
                    with self._peer.lock:
                        peer = self._peer.connect(
                            str(ready["ucx_metadata"]),
                            str(ready["ucx_agent_name"]),
                        )
                        self._peer.transfer(
                            "WRITE" if operation == "store" else "READ",
                            staging.address,
                            int(ready["ucx_address"]),
                            length,
                            peer,
                        )
                        self._transfer_bytes += length
                    _send_frame(
                        connection,
                        {"version": _PROTOCOL_VERSION, "phase": "transferred"},
                        b"",
                    )
                    terminal, terminal_length = _recv_header(connection)
                    if terminal_length or terminal.get("version") != _PROTOCOL_VERSION:
                        raise TransportError("DATA_LOSS", "invalid UCX terminal frame")
                    if terminal.get("error_status"):
                        raise TransportError(
                            str(terminal["error_status"]),
                            str(terminal.get("detail", "")),
                        )
                    results = terminal.get("results")
                    if not isinstance(results, list) or len(results) != len(items):
                        raise TransportError("DATA_LOSS", "invalid UCX result count")
                    return results, staging.read(length) if operation == "load" else b""
            except UcxOwnershipError:
                retain_staging = True
                raise
            except (TimeoutError, socket.timeout) as exc:
                raise TransportError(
                    "DEADLINE_EXCEEDED", "UCX control timed out"
                ) from exc
            except OSError as exc:
                raise TransportError(
                    "UNAVAILABLE", f"UCX control failed: {exc}"
                ) from exc
        finally:
            if staging is not None:
                self._pool.release(staging, uncertain=retain_staging)

    def close(self) -> None:
        self._pool.close()


class UcxShardServer(TcpShardServer):
    """Keep server staging registered until client RDMA and file I/O complete."""

    def __init__(
        self,
        endpoint: TcpEndpoint,
        handler: Callable,
        agent: Any,
        *,
        terminal_callback: Callable,
        timeout: float,
        max_request_bytes: int,
        max_staging_bytes: int,
        max_workers: int,
    ):
        self._peer = _UcxPeer(agent)
        self._terminal_callback = terminal_callback
        self._pool = _StagingPool(agent, max_staging_bytes)
        super().__init__(
            endpoint,
            handler,
            timeout=timeout,
            max_request_bytes=max_request_bytes,
            max_staging_bytes=max_staging_bytes,
            max_workers=max_workers,
            allow_remote=True,
        )

    def _handle(self, connection: socket.socket) -> None:
        length = 0
        staging = None
        transfer_pending = False
        try:
            header, payload_length = _recv_header(connection)
            if header.get("version") != _PROTOCOL_VERSION or payload_length:
                raise TransportError("DATA_LOSS", "invalid UCX request frame")
            operation = header.get("operation")
            if operation not in ("load", "store"):
                raise TransportError("INVALID_ARGUMENT", "unknown UCX operation")
            items, length = self._validate_items(header.get("items"))
            if length > self._max_request_bytes:
                raise TransportError(
                    "RESOURCE_EXHAUSTED", "UCX server staging budget exhausted"
                )
            staging = self._pool.acquire(length, self._timeout)
            if operation == "load":
                results, data = self._handler(operation, items, b"")
                staging.write(data)
            # A lost control connection does not prove that an RDMA WRITE
            # stopped touching this address. Retain its registration and
            # budget until process teardown if completion is ambiguous.
            transfer_pending = True
            connection.settimeout(None)
            _send_frame(
                connection,
                {
                    "version": _PROTOCOL_VERSION,
                    "ucx_metadata": self._peer.metadata(),
                    "ucx_agent_name": self._peer.agent.name,
                    "ucx_address": staging.address,
                },
                b"",
            )
            transferred, transferred_length = _recv_header(connection)
            if (
                transferred_length
                or transferred.get("phase") != "transferred"
                or transferred.get("version") != _PROTOCOL_VERSION
            ):
                raise TransportError("DATA_LOSS", "invalid UCX transfer completion")
            transfer_pending = False
            if operation == "store":
                results, _ = self._handler(operation, items, staging.read(length))
            self._terminal_callback(items)
            _send_frame(
                connection, {"version": _PROTOCOL_VERSION, "results": results}, b""
            )
        except TransportError as exc:
            try:
                _send_frame(
                    connection,
                    {
                        "version": _PROTOCOL_VERSION,
                        "error_status": exc.status,
                        "detail": exc.detail,
                    },
                    b"",
                )
            except OSError:
                pass
        except Exception as exc:
            try:
                _send_frame(
                    connection,
                    {
                        "version": _PROTOCOL_VERSION,
                        "error_status": "INTERNAL",
                        "detail": str(exc),
                    },
                    b"",
                )
            except OSError:
                pass
        finally:
            if staging is not None:
                self._pool.release(staging, uncertain=transfer_pending)

    def close(self, timeout: float | None = None) -> None:
        super().close(timeout)
        self._pool.close()
