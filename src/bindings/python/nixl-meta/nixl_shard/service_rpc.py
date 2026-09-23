# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small framed-JSON TCP transport for :mod:`nixl_shard.service`.

This is a functional V1 control transport, not a public compatibility
boundary.  Frames are bounded and each connection carries one request, which
keeps failure and timeout behavior easy to reason about while metadata calls
remain batched.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import ipaddress
import json
import socket
import socketserver
import struct
import threading
from typing import Any, Optional, Sequence

from . import service as service_types
from .service import (
    AbortItem,
    BatchRejectedError,
    CommitItem,
    DeviceRegistration,
    IoTerminalReport,
    LookupItem,
    ReleaseReadItem,
    RenewReadItem,
    ReserveItem,
    ShardNamingService,
    TouchItem,
)

_HEADER = struct.Struct("!I")
_DEFAULT_MAX_FRAME = 4 << 20


def _is_loopback_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(addresses) and all(
        ipaddress.ip_address(address[4][0]).is_loopback for address in addresses
    )


_RPC_METHODS = {
    "register_device",
    "heartbeat",
    "list_devices",
    "get_device_status",
    "stop_device",
    "drain_device",
    "batch_reserve",
    "batch_commit",
    "batch_abort",
    "batch_lookup",
    "batch_exists",
    "batch_renew_read",
    "batch_release_read",
    "batch_touch",
    "batch_report_io_terminal",
    "report_io_terminal",
    "set_watermarks",
    "run_eviction",
    "get_capacity",
    "reap_expired",
    "fence_device_epoch",
    "restart",
    "health",
    "get_object_debug",
    "get_metrics",
    "validate_invariants",
}


class RemoteServiceError(RuntimeError):
    def __init__(self, error_type: str, message: str):
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


def _type_registry() -> dict[str, type]:
    registry: dict[str, type] = {}
    for name in dir(service_types):
        value = getattr(service_types, name)
        if isinstance(value, type) and (
            dataclasses.is_dataclass(value) or issubclass(value, enum.Enum)
        ):
            registry[name] = value
    return registry


_TYPES = _type_registry()


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            "fields": {
                field.name: _encode(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, enum.Enum):
        return {"__enum__": type(value).__name__, "value": value.value}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, set):
        return {"__set__": [_encode(item) for item in sorted(value, key=repr)]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot encode RPC value of type {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__bytes__" in value:
        return base64.b64decode(value["__bytes__"], validate=True)
    if "__tuple__" in value:
        return tuple(_decode(item) for item in value["__tuple__"])
    if "__set__" in value:
        return {_decode(item) for item in value["__set__"]}
    if "__enum__" in value:
        enum_type = _TYPES.get(value["__enum__"])
        if enum_type is None or not issubclass(enum_type, enum.Enum):
            raise ValueError("unknown enum in RPC payload")
        return enum_type(value["value"])
    if "__type__" in value:
        data_type = _TYPES.get(value["__type__"])
        if data_type is None or not dataclasses.is_dataclass(data_type):
            raise ValueError("unknown dataclass in RPC payload")
        fields = value.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("invalid dataclass fields in RPC payload")
        return data_type(**{name: _decode(item) for name, item in fields.items()})
    return {key: _decode(item) for key, item in value.items()}


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("peer closed an incomplete RPC frame")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_frame(sock: socket.socket, max_frame_bytes: int) -> Any:
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if length <= 0 or length > max_frame_bytes:
        raise ValueError("RPC frame exceeds configured limit")
    return _decode(json.loads(_recv_exact(sock, length).decode("utf-8")))


def _send_frame(sock: socket.socket, value: Any, max_frame_bytes: int) -> None:
    payload = json.dumps(_encode(value), separators=(",", ":")).encode("utf-8")
    if len(payload) > max_frame_bytes:
        raise ValueError("RPC frame exceeds configured limit")
    sock.sendall(_HEADER.pack(len(payload)) + payload)


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        self.request.settimeout(server.request_timeout)
        try:
            request = _recv_frame(self.request, server.max_frame_bytes)
            if not isinstance(request, dict):
                raise ValueError("RPC request must be an object")
            method = request.get("method")
            if method not in _RPC_METHODS:
                raise ValueError("unknown RPC method")
            args = request.get("args", [])
            kwargs = request.get("kwargs", {})
            if not isinstance(args, list) or not isinstance(kwargs, dict):
                raise ValueError("invalid RPC arguments")
            result = getattr(server.naming_service, method)(*args, **kwargs)
            response = {"ok": True, "result": result}
        except Exception as exc:  # Transport preserves typed service failures.
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        try:
            _send_frame(self.request, response, server.max_frame_bytes)
        except (ConnectionError, OSError):
            pass


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address,
        naming_service,
        max_frame_bytes,
        request_timeout,
        max_workers,
    ):
        self.naming_service = naming_service
        self.max_frame_bytes = max_frame_bytes
        self.request_timeout = request_timeout
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        super().__init__(address, _RequestHandler)

    def process_request(self, request, client_address):
        if not self._worker_slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()


class ShardNamingTCPServer:
    """Threaded TCP endpoint for a :class:`ShardNamingService`."""

    def __init__(
        self,
        service: ShardNamingService,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME,
        request_timeout: float = 5.0,
        max_workers: int = 32,
    ):
        if max_frame_bytes <= 0 or request_timeout <= 0 or max_workers <= 0:
            raise ValueError("RPC limits and timeout must be positive")
        if not _is_loopback_host(host):
            raise ValueError("naming TCP server must bind to a loopback address")
        self._server = _ThreadingTCPServer(
            (host, port),
            service,
            max_frame_bytes,
            request_timeout,
            max_workers,
        )
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> "ShardNamingTCPServer":
        if self._thread is not None:
            raise RuntimeError("server is already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="nixl-shard-naming",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._server.server_close()

    def __enter__(self) -> "ShardNamingTCPServer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ShardNamingTCPClient:
    """Blocking loopback TCP proxy with the same batched service methods."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float = 5.0,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME,
        max_batch_items: int = 256,
        max_batch_bytes: int = 1 << 20,
    ):
        if (
            timeout <= 0
            or max_frame_bytes <= 0
            or max_batch_items <= 0
            or max_batch_bytes <= 0
        ):
            raise ValueError("timeouts and limits must be positive")
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_frame_bytes = max_frame_bytes
        self.max_batch_items = max_batch_items
        self.max_batch_bytes = max_batch_bytes

    def _call(self, method: str, *args, **kwargs):
        if method not in _RPC_METHODS:
            raise ValueError("unknown RPC method")
        with socket.create_connection((self.host, self.port), self.timeout) as sock:
            sock.settimeout(self.timeout)
            _send_frame(
                sock,
                {"method": method, "args": list(args), "kwargs": kwargs},
                self.max_frame_bytes,
            )
            response = _recv_frame(sock, self.max_frame_bytes)
        if not response.get("ok"):
            raise RemoteServiceError(
                response.get("error_type", "RemoteError"),
                response.get("message", "remote service failed"),
            )
        return response.get("result")

    def register_device(self, registration: DeviceRegistration):
        return self._call("register_device", registration)

    def heartbeat(self, device_id: int, agent_epoch: int):
        return self._call("heartbeat", device_id, agent_epoch)

    def list_devices(self):
        return self._call("list_devices")

    def get_device_status(self, device_id: int):
        return self._call("get_device_status", device_id)

    def stop_device(
        self,
        device_id: int,
        agent_epoch: int,
        *,
        device_generation: Optional[int] = None,
        grace_period_s: Optional[float] = None,
    ):
        return self._call(
            "stop_device",
            device_id,
            agent_epoch,
            device_generation=device_generation,
            grace_period_s=grace_period_s,
        )

    def drain_device(
        self,
        device_id: int,
        agent_epoch: int,
        *,
        device_generation: Optional[int] = None,
        grace_period_s: Optional[float] = None,
    ):
        return self._call(
            "drain_device",
            device_id,
            agent_epoch,
            device_generation=device_generation,
            grace_period_s=grace_period_s,
        )

    def batch_reserve(self, items: Sequence[ReserveItem], **kwargs):
        return self._call("batch_reserve", list(items), **kwargs)

    def batch_commit(self, items: Sequence[CommitItem], **kwargs):
        return self._call("batch_commit", list(items), **kwargs)

    def batch_abort(self, items: Sequence[AbortItem], **kwargs):
        return self._call("batch_abort", list(items), **kwargs)

    def batch_lookup(self, items: Sequence[LookupItem], **kwargs):
        return self._call("batch_lookup", list(items), **kwargs)

    def batch_exists(self, keys: Sequence[bytes], **kwargs):
        return self._call("batch_exists", list(keys), **kwargs)

    def batch_renew_read(self, items: Sequence[RenewReadItem], **kwargs):
        return self._call("batch_renew_read", list(items), **kwargs)

    def batch_release_read(self, items: Sequence[ReleaseReadItem], **kwargs):
        return self._call("batch_release_read", list(items), **kwargs)

    def batch_touch(self, items: Sequence[TouchItem], **kwargs):
        return self._call("batch_touch", list(items), **kwargs)

    def batch_report_io_terminal(self, items: Sequence[IoTerminalReport], **kwargs):
        return self._call("batch_report_io_terminal", list(items), **kwargs)

    def report_io_terminal(self, item: IoTerminalReport):
        return self._call("report_io_terminal", item)

    def set_watermarks(self, device_id: int, low: float, high: float):
        return self._call("set_watermarks", device_id, low, high)

    def run_eviction(self, device_id: Optional[int] = None, *, force: bool = False):
        return self._call("run_eviction", device_id, force=force)

    def get_capacity(self, device_id: Optional[int] = None):
        return self._call("get_capacity", device_id)

    def reap_expired(self):
        return self._call("reap_expired")

    def fence_device_epoch(
        self,
        device_id: int,
        agent_epoch: int,
        *,
        quiesced: bool,
        device_generation: Optional[int] = None,
    ):
        return self._call(
            "fence_device_epoch",
            device_id,
            agent_epoch,
            quiesced=quiesced,
            device_generation=device_generation,
        )

    def restart(self):
        return self._call("restart")

    def health(self):
        return self._call("health")

    def get_object_debug(self, key_or_digest):
        return self._call("get_object_debug", key_or_digest)

    def get_metrics(self):
        return self._call("get_metrics")

    def validate_invariants(self):
        return self._call("validate_invariants")
