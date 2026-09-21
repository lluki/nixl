# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded framed TCP transport for the functional remote data plane."""

from __future__ import annotations

import ipaddress
import json
import socket
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

_PROTOCOL_VERSION = 1
_MAX_HEADER_BYTES = 64 * 1024


def _is_loopback_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(addresses) and all(
        ipaddress.ip_address(address[4][0]).is_loopback for address in addresses
    )


@dataclass(frozen=True)
class TcpEndpoint:
    host: str
    port: int


@dataclass(frozen=True)
class RemoteDevice:
    logical_device_id: int
    host: str
    port: int
    logical_device_generation: int = 1

    @property
    def endpoint(self) -> TcpEndpoint:
        return TcpEndpoint(self.host, self.port)


class TransportError(RuntimeError):
    """A transport failure with a public completion-status name."""

    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class _StagingBudget:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._available = capacity
        self._condition = threading.Condition()

    def acquire(self, amount: int, timeout: float) -> bool:
        if amount > self._capacity:
            return False
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._available >= amount, timeout=timeout
            )
            if ready:
                self._available -= amount
            return ready

    def release(self, amount: int) -> None:
        with self._condition:
            self._available += amount
            self._condition.notify_all()


def _json_bytes(header: dict[str, Any]) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_HEADER_BYTES:
        raise TransportError("INVALID_ARGUMENT", "transport header is too large")
    return encoded


def _send_frame(sock: socket.socket, header: dict[str, Any], payload: bytes) -> None:
    header = dict(header)
    header["payload_length"] = len(payload)
    encoded = _json_bytes(header)
    sock.sendall(struct.pack("!I", len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise TransportError("UNAVAILABLE", "peer disconnected during a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_header(sock: socket.socket) -> tuple[dict[str, Any], int]:
    (header_length,) = struct.unpack("!I", _recv_exact(sock, 4))
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise TransportError("DATA_LOSS", "invalid transport header length")
    try:
        header = json.loads(_recv_exact(sock, header_length))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransportError("DATA_LOSS", f"invalid transport header: {exc}") from exc
    if not isinstance(header, dict):
        raise TransportError("DATA_LOSS", "transport header is not an object")
    payload_length = header.get("payload_length")
    if not isinstance(payload_length, int) or payload_length < 0:
        raise TransportError("DATA_LOSS", "invalid payload length")
    return header, payload_length


class TcpShardTransport:
    """One-request-per-connection client with a shared staging-byte budget."""

    def __init__(
        self,
        *,
        timeout: float,
        max_request_bytes: int,
        max_staging_bytes: int,
    ):
        self._timeout = timeout
        self._max_request_bytes = max_request_bytes
        self._budget = _StagingBudget(max_staging_bytes)

    def request(
        self,
        endpoint: TcpEndpoint,
        operation: str,
        items: Sequence[dict[str, Any]],
        payload: bytes | Callable[[], bytes],
    ) -> tuple[list[dict[str, Any]], bytes]:
        staged_bytes = sum(int(item["length"]) for item in items)
        if staged_bytes > self._max_request_bytes:
            raise TransportError(
                "RESOURCE_EXHAUSTED", "remote batch exceeds maximum request bytes"
            )
        if operation == "load" and payload:
            raise TransportError("INTERNAL", "load request unexpectedly has a payload")
        if not self._budget.acquire(staged_bytes, self._timeout):
            raise TransportError(
                "RESOURCE_EXHAUSTED", "client staging budget exhausted"
            )

        try:
            if callable(payload):
                payload = payload()
            if operation == "store" and len(payload) != staged_bytes:
                raise TransportError(
                    "INTERNAL", "store payload length does not match items"
                )
            header = {
                "version": _PROTOCOL_VERSION,
                "operation": operation,
                "items": list(items),
            }
            try:
                with socket.create_connection(
                    (endpoint.host, endpoint.port), timeout=self._timeout
                ) as connection:
                    # Once request bytes can reach the peer, a timeout is
                    # ambiguous: the peer may still complete fixed-file I/O.
                    # Wait for its terminal reply or a definitive disconnect.
                    connection.settimeout(None)
                    _send_frame(connection, header, payload)
                    response, response_length = _recv_header(connection)
                    if response.get("version") != _PROTOCOL_VERSION:
                        raise TransportError(
                            "DATA_LOSS", "unsupported response version"
                        )
                    error_status = response.get("error_status")
                    if error_status:
                        if response_length:
                            raise TransportError(
                                "DATA_LOSS", "error response unexpectedly has a payload"
                            )
                        raise TransportError(
                            str(error_status), str(response.get("detail", ""))
                        )
                    expected_payload = staged_bytes if operation == "load" else 0
                    if response_length != expected_payload:
                        raise TransportError(
                            "DATA_LOSS",
                            "response payload length does not match batch",
                        )
                    response_payload = _recv_exact(connection, response_length)
            except (TimeoutError, socket.timeout) as exc:
                raise TransportError(
                    "DEADLINE_EXCEEDED", "remote request timed out"
                ) from exc
            except OSError as exc:
                raise TransportError(
                    "UNAVAILABLE", f"remote connection failed: {exc}"
                ) from exc

            results = response.get("results")
            if not isinstance(results, list) or len(results) != len(items):
                raise TransportError(
                    "DATA_LOSS", "response result count does not match batch"
                )
            return results, response_payload
        finally:
            self._budget.release(staged_bytes)


class TcpShardServer:
    """Bounded TCP server for one logical-device handler."""

    def __init__(
        self,
        endpoint: TcpEndpoint,
        handler: Callable[
            [str, Sequence[dict[str, Any]], bytes],
            tuple[list[dict[str, Any]], bytes],
        ],
        *,
        timeout: float,
        max_request_bytes: int,
        max_staging_bytes: int,
        max_workers: int,
    ):
        if not _is_loopback_host(endpoint.host):
            raise ValueError("TCP fallback server must bind to a loopback address")
        self._handler = handler
        self._timeout = timeout
        self._max_request_bytes = max_request_bytes
        self._budget = _StagingBudget(max_staging_bytes)
        self._closed = threading.Event()
        self._worker_slots = threading.BoundedSemaphore(max_workers)
        self._futures_lock = threading.Lock()
        self._futures: set[Future[Any]] = set()
        self._executor_closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="nixlshard-tcp"
        )
        family = socket.AF_INET6 if ":" in endpoint.host else socket.AF_INET
        self._socket = socket.socket(family, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((endpoint.host, endpoint.port))
        self._socket.listen(max_workers)
        self._socket.settimeout(0.1)
        bound = self._socket.getsockname()
        self.endpoint = TcpEndpoint(str(bound[0]), int(bound[1]))
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="nixlshard-tcp-accept", daemon=True
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                connection, _ = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            connection.settimeout(self._timeout)
            if not self._worker_slots.acquire(blocking=False):
                try:
                    _send_frame(
                        connection,
                        {
                            "version": _PROTOCOL_VERSION,
                            "error_status": "RESOURCE_EXHAUSTED",
                            "detail": "server worker limit reached",
                        },
                        b"",
                    )
                except OSError:
                    pass
                connection.close()
                continue
            try:
                future = self._executor.submit(self._handle_and_release, connection)
            except RuntimeError:
                self._worker_slots.release()
                connection.close()
                continue
            with self._futures_lock:
                self._futures.add(future)
            future.add_done_callback(self._worker_done)

    def _worker_done(self, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def _handle_and_release(self, connection: socket.socket) -> None:
        try:
            with connection:
                self._handle(connection)
        finally:
            self._worker_slots.release()

    @staticmethod
    def _validate_items(items: Any) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(items, list) or not items:
            raise TransportError("INVALID_ARGUMENT", "remote batch must contain items")
        total = 0
        for item in items:
            if not isinstance(item, dict):
                raise TransportError("INVALID_ARGUMENT", "remote item is not an object")
            required = (
                "device_id",
                "device_generation",
                "device_offset",
                "length",
                "request_id",
            )
            if any(field not in item for field in required):
                raise TransportError(
                    "INVALID_ARGUMENT", "remote item is missing fields"
                )
            if not all(isinstance(item[field], int) for field in required[:4]):
                raise TransportError(
                    "INVALID_ARGUMENT", "remote item has invalid integers"
                )
            if item["device_generation"] <= 0:
                raise TransportError(
                    "INVALID_ARGUMENT", "remote device generation is invalid"
                )
            authority_epoch = item.get("authority_epoch")
            if authority_epoch is not None and (
                not isinstance(authority_epoch, int) or authority_epoch <= 0
            ):
                raise TransportError(
                    "INVALID_ARGUMENT", "remote authority epoch is invalid"
                )
            terminal_report = item.get("terminal_report")
            if terminal_report is not None and not isinstance(terminal_report, dict):
                raise TransportError(
                    "INVALID_ARGUMENT", "remote terminal report is invalid"
                )
            if not isinstance(item["request_id"], str) or len(item["request_id"]) > 128:
                raise TransportError(
                    "INVALID_ARGUMENT", "remote item has an invalid request id"
                )
            if item["length"] <= 0:
                raise TransportError(
                    "INVALID_ARGUMENT", "remote item length is invalid"
                )
            total += item["length"]
        return items, total

    def _handle(self, connection: socket.socket) -> None:
        staged_bytes = 0
        budget_acquired = False
        try:
            header, payload_length = _recv_header(connection)
            if header.get("version") != _PROTOCOL_VERSION:
                raise TransportError("DATA_LOSS", "unsupported request version")
            operation = header.get("operation")
            if operation not in ("load", "store"):
                raise TransportError("INVALID_ARGUMENT", "unknown remote operation")
            items, staged_bytes = self._validate_items(header.get("items"))
            if staged_bytes > self._max_request_bytes:
                raise TransportError(
                    "RESOURCE_EXHAUSTED", "remote batch exceeds maximum request bytes"
                )
            expected_payload = staged_bytes if operation == "store" else 0
            if payload_length != expected_payload:
                raise TransportError(
                    "DATA_LOSS", "request payload length does not match batch"
                )
            budget_acquired = self._budget.acquire(staged_bytes, self._timeout)
            if not budget_acquired:
                raise TransportError(
                    "RESOURCE_EXHAUSTED", "server staging budget exhausted"
                )
            payload = _recv_exact(connection, payload_length)
            results, response_payload = self._handler(operation, items, payload)
            _send_frame(
                connection,
                {"version": _PROTOCOL_VERSION, "results": results},
                response_payload,
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
        except (TimeoutError, socket.timeout):
            try:
                _send_frame(
                    connection,
                    {
                        "version": _PROTOCOL_VERSION,
                        "error_status": "DEADLINE_EXCEEDED",
                        "detail": "server request timed out",
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
            if budget_acquired:
                self._budget.release(staged_bytes)

    def close(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        if not self._closed.is_set():
            self._closed.set()
            self._socket.close()
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        self._accept_thread.join(timeout=remaining)
        if self._accept_thread.is_alive():
            raise TimeoutError("TCP server accept thread did not stop")

        while True:
            with self._futures_lock:
                active = bool(self._futures)
            if not active:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("TCP server still has active requests")
            time.sleep(0.001)

        if not self._executor_closed:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor_closed = True
