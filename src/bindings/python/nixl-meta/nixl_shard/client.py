# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process, batch-only KV client for NIXLShard."""

from __future__ import annotations

import hashlib
import inspect
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from .agent import (
    BufferRegion,
    CompletionStatus,
    IoItem,
    OperationTimeoutError,
    ShardAgent,
    buffer_region_from_writable,
)
from .service import (
    AbortItem,
    CommitItem,
    IoTerminalReport,
    IOTokenKind,
    LookupItem,
    ReleaseReadItem,
    ReserveItem,
    TouchItem,
)
from .trace import emit_trace


class KVStatus(str, Enum):
    OK = "OK"
    MISS = "MISS"
    ALREADY_PRESENT = "ALREADY_PRESENT"
    CANCELLED = "CANCELLED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    IO_ERROR = "IO_ERROR"
    DATA_LOSS = "DATA_LOSS"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True)
class ClientConfig:
    service: Any
    agent: ShardAgent
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    default_timeout_s: float = 30.0
    reservation_ttl_s: float = 30.0
    read_lease_ttl_s: float = 30.0
    max_inflight_batches: int = 16
    close_agent_on_close: bool = False
    heartbeat_device_id: int | None = None
    heartbeat_agent_epoch: int | None = None
    heartbeat_device_generation: int | None = None
    heartbeat_interval_s: float = 10.0


@dataclass(frozen=True)
class SetItem:
    key: bytes | str
    source: Any
    length: int | None = None
    preferred_device_id: int | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class GetItem:
    key: bytes | str
    destination: Any
    capacity: int | None = None
    preferred_device_id: int | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True)
class KVResult:
    key: bytes
    status: KVStatus
    logical_length: int = 0
    detail: str = ""
    device_id: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in (KVStatus.OK, KVStatus.ALREADY_PRESENT)


@dataclass(frozen=True)
class ClientOperationHandle:
    id: str


@dataclass
class _AsyncBatch:
    future: Future[list[KVResult]]
    handles: tuple[ClientOperationHandle, ...]
    consumed: set[int] = field(default_factory=set)


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        if not key:
            raise ValueError("key must not be empty")
        return key
    if isinstance(key, str):
        encoded = key.encode("utf-8")
        if not encoded:
            raise ValueError("key must not be empty")
        return encoded
    raise TypeError("key must be bytes or str")


def _key_digest(key: bytes | str) -> str:
    return hashlib.sha256(_key_bytes(key)).hexdigest()


def _buffer_region(value: Any) -> BufferRegion:
    if isinstance(value, BufferRegion):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        address, length = value
        return BufferRegion(int(address), int(length), owner=value)
    return buffer_region_from_writable(value)


def _buffer_regions(value: Any) -> tuple[BufferRegion, ...]:
    """Normalize one contiguous buffer or a scatter/gather buffer list."""

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(part, int) for part in value)
    ):
        return (_buffer_region(value),)
    if (
        isinstance(value, (list, tuple))
        and value
        and all(
            isinstance(part, BufferRegion)
            or (
                isinstance(part, tuple)
                and len(part) == 2
                and all(isinstance(field, int) for field in part)
            )
            for part in value
        )
    ):
        return tuple(_buffer_region(part) for part in value)
    return (_buffer_region(value),)


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _result_code(value: Any) -> str:
    return _enum_name(getattr(value, "code", getattr(value, "status", value)))


def _make(cls: type[Any], **values: Any) -> Any:
    """Construct service dataclasses while tolerating optional wire-only fields."""

    parameters = inspect.signature(cls).parameters
    return cls(**{key: value for key, value in values.items() if key in parameters})


def _chunks(
    values: Sequence[Any], max_items: int, max_bytes: int
) -> Iterable[Sequence[Any]]:
    chunk: list[Any] = []
    encoded_bytes = 0
    for item in values:
        item_bytes = len(repr(item).encode("utf-8"))
        if chunk and (
            len(chunk) >= max_items or encoded_bytes + item_bytes > max_bytes
        ):
            yield chunk
            chunk = []
            encoded_bytes = 0
        chunk.append(item)
        encoded_bytes += item_bytes
    if chunk:
        yield chunk


class ShardClient:
    """Compose shardnserv metadata and the in-process shardagent data path."""

    def __init__(self, config: ClientConfig):
        if config.default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be positive")
        if config.reservation_ttl_s <= 0 or config.read_lease_ttl_s <= 0:
            raise ValueError("lease durations must be positive")
        if config.max_inflight_batches <= 0:
            raise ValueError("max_inflight_batches must be positive")
        if (config.heartbeat_device_id is None) != (
            config.heartbeat_agent_epoch is None
        ):
            raise ValueError("heartbeat device ID and agent epoch must be set together")
        if config.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        self._config = config
        self._service = config.service
        self._agent = config.agent
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_inflight_batches,
            thread_name_prefix="nixlshard-client",
        )
        self._async_admission = threading.BoundedSemaphore(config.max_inflight_batches)
        self._lock = threading.RLock()
        self._async: dict[str, tuple[_AsyncBatch, int]] = {}
        self._closed = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._metrics = {
            "set_items": 0,
            "get_items": 0,
            "exists_items": 0,
            "hits": 0,
            "misses": 0,
            "metadata_calls": 0,
            "data_bytes": 0,
            "heartbeat_successes": 0,
            "heartbeat_failures": 0,
        }
        if config.heartbeat_device_id is not None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"nixlshard-heartbeat-{config.heartbeat_device_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @property
    def client_id(self) -> str:
        return self._config.client_id

    def batch_set(
        self,
        items_or_keys: Sequence[SetItem] | Sequence[bytes | str],
        sources: Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> list[KVResult]:
        self._ensure_open()
        items = self._coerce_set_items(items_or_keys, sources)
        if not items:
            return []
        timeout = self._config.default_timeout_s if timeout is None else timeout
        started = time.monotonic()
        results: list[KVResult | None] = [None] * len(items)
        valid: list[tuple[int, SetItem, bytes, tuple[BufferRegion, ...], int]] = []
        for index, item in enumerate(items):
            try:
                key = _key_bytes(item.key)
                regions = _buffer_regions(item.source)
                available = sum(region.length for region in regions)
                length = available if item.length is None else item.length
                if length <= 0 or length > available:
                    raise ValueError("length exceeds source buffer")
                if len(regions) > 1 and length != available:
                    raise ValueError("scatter/gather length must cover every region")
                valid.append((index, item, key, regions, length))
            except (TypeError, ValueError) as exc:
                key = (
                    item.key if isinstance(item.key, bytes) else str(item.key).encode()
                )
                results[index] = KVResult(
                    key, KVStatus.INVALID_ARGUMENT, detail=str(exc)
                )

        reserve_items = [
            _make(
                ReserveItem,
                key=key,
                logical_length=length,
                preferred_device_id=item.preferred_device_id,
                reservation_ttl_s=self._config.reservation_ttl_s,
                reservation_ttl_ms=int(self._config.reservation_ttl_s * 1000),
                client_id=self.client_id,
                request_id=item.request_id,
            )
            for _, item, key, _, length in valid
        ]
        reserve_results = self._metadata("batch_reserve", reserve_items)

        pending: list[tuple[int, bytes, tuple[BufferRegion, ...], Any]] = []
        for (index, _, key, regions, length), reserve in zip(
            valid, reserve_results, strict=True
        ):
            code = _result_code(reserve)
            reservation = getattr(reserve, "reservation", None)
            mapping = getattr(reserve, "mapping", None)
            if code in {"ALREADY_PRESENT", "EXISTING_READY", "READY"} or (
                reservation is None and mapping is not None
            ):
                results[index] = KVResult(
                    key,
                    KVStatus.ALREADY_PRESENT,
                    logical_length=getattr(mapping, "logical_length", length),
                )
            elif reservation is not None and code in {"OK", "RESERVED", "SUCCESS"}:
                pending.append((index, key, regions, reservation))
            else:
                results[index] = KVResult(
                    key,
                    self._metadata_failure(code, is_lookup=False),
                    detail=getattr(reserve, "detail", code),
                )

        if pending:
            flat_regions = [region for entry in pending for region in entry[2]]
            try:
                memory = self._agent.register_memory(flat_regions)
            except BaseException:
                self._abandon_reservations([entry[3] for entry in pending])
                raise
            completions = []
            handles: Sequence[Any] = ()
            groups: list[tuple[int, int]] = []
            try:
                io_items = []
                memory_index = 0
                for _, _, regions, reservation in pending:
                    start = len(io_items)
                    device_offset = reservation.offset
                    remaining = reservation.logical_length
                    for region in regions:
                        length = min(region.length, remaining)
                        io_items.append(
                            IoItem(
                                reservation.device_id,
                                device_offset,
                                memory,
                                memory_index,
                                0,
                                length,
                                device_generation=getattr(
                                    reservation,
                                    "device_generation",
                                    reservation.agent_epoch,
                                ),
                                authority_epoch=getattr(
                                    reservation, "service_epoch", None
                                ),
                                terminal_report=self._terminal_report_dict(reservation),
                            )
                        )
                        memory_index += 1
                        device_offset += length
                        remaining -= length
                    groups.append((start, len(io_items) - start))
                handles = self._agent.submit_batch_store(io_items)
                completions = self._safe_wait(handles, started, timeout)
            except BaseException:
                terminal = not handles
                if handles:
                    try:
                        self._agent.wait(handles, 0)
                        terminal = True
                    except BaseException:
                        terminal = False
                tokens = [entry[3] for entry in pending]
                if terminal:
                    self._agent.unregister_memory(memory)
                    self._abandon_reservations(tokens)
                else:
                    self._defer_agent_cleanup(handles, memory, tokens, is_write=True)
                raise
            finally:
                if completions:
                    self._agent.unregister_memory(memory)

            reports = []
            commits = []
            aborts = []
            completion_by_index: dict[int, tuple[bool, CompletionStatus, str]] = {}
            for entry, (start, count) in zip(pending, groups, strict=True):
                index, key, _, reservation = entry
                item_completions = completions[start : start + count]
                failure = next(
                    (
                        completion
                        for completion in item_completions
                        if completion.status is not CompletionStatus.OK
                    ),
                    None,
                )
                success = failure is None
                status = CompletionStatus.OK if success else failure.status
                detail = "" if success else failure.detail
                completion_by_index[index] = (success, status, detail)
                reports.append(self._terminal_report(reservation))
                common = dict(
                    reservation_id=getattr(reservation, "reservation_id", None),
                    generation=getattr(reservation, "generation", None),
                    service_epoch=getattr(reservation, "service_epoch", None),
                    client_id=self.client_id,
                    request_id=uuid.uuid4().hex,
                )
                if success:
                    commits.append(_make(CommitItem, **common))
                else:
                    aborts.append(_make(AbortItem, **common))
                    results[index] = KVResult(
                        key,
                        self._completion_status(status),
                        detail=detail,
                    )
            self._report_terminal(reports)
            commit_results = self._metadata("batch_commit", commits) if commits else []
            if aborts:
                self._metadata("batch_abort", aborts)
            commit_iter = iter(commit_results)
            failed_commits = []
            for index, key, regions, reservation in pending:
                success, _, _ = completion_by_index[index]
                if not success:
                    continue
                commit = next(commit_iter)
                code = _result_code(commit)
                if code in {"OK", "SUCCESS", "ALREADY_PRESENT", "READY"}:
                    results[index] = KVResult(
                        key,
                        KVStatus.OK,
                        logical_length=getattr(
                            reservation,
                            "logical_length",
                            sum(region.length for region in regions),
                        ),
                        device_id=getattr(reservation, "device_id", None),
                    )
                else:
                    failed_commits.append(
                        _make(
                            AbortItem,
                            reservation_id=reservation.reservation_id,
                            generation=reservation.generation,
                            service_epoch=reservation.service_epoch,
                            client_id=self.client_id,
                            request_id=uuid.uuid4().hex,
                        )
                    )
                    results[index] = KVResult(
                        key,
                        self._metadata_failure(code, is_lookup=False),
                        detail=getattr(commit, "detail", code),
                    )
            if failed_commits:
                self._metadata("batch_abort", failed_commits)

        final = [result for result in results if result is not None]
        assert len(final) == len(items)
        with self._lock:
            self._metrics["set_items"] += len(items)
            self._metrics["data_bytes"] += sum(
                result.logical_length for result in final if result.ok
            )
        return final

    def batch_get(
        self,
        items_or_keys: Sequence[GetItem] | Sequence[bytes | str],
        destinations: Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> list[KVResult]:
        self._ensure_open()
        items = self._coerce_get_items(items_or_keys, destinations)
        if not items:
            return []
        timeout = self._config.default_timeout_s if timeout is None else timeout
        started = time.monotonic()
        started_ns = time.perf_counter_ns()
        results: list[KVResult | None] = [None] * len(items)
        valid: list[tuple[int, GetItem, bytes, tuple[BufferRegion, ...], int]] = []
        for index, item in enumerate(items):
            try:
                key = _key_bytes(item.key)
                regions = _buffer_regions(item.destination)
                available = sum(region.length for region in regions)
                capacity = available if item.capacity is None else item.capacity
                if capacity <= 0 or capacity > available:
                    raise ValueError("capacity exceeds destination buffer")
                if len(regions) > 1 and capacity != available:
                    raise ValueError("scatter/gather capacity must cover every region")
                valid.append((index, item, key, regions, capacity))
            except (TypeError, ValueError) as exc:
                key = (
                    item.key if isinstance(item.key, bytes) else str(item.key).encode()
                )
                results[index] = KVResult(
                    key, KVStatus.INVALID_ARGUMENT, detail=str(exc)
                )

        lookups = [
            _make(
                LookupItem,
                key=key,
                preferred_device_id=item.preferred_device_id,
                lease_ttl_s=self._config.read_lease_ttl_s,
                client_id=self.client_id,
                request_id=item.request_id,
            )
            for _, item, key, _, _ in valid
        ]
        lookup_started_ns = time.perf_counter_ns()
        lookup_results = self._metadata("batch_lookup", lookups)
        lookup_ended_ns = time.perf_counter_ns()
        lookup_ns = lookup_ended_ns - lookup_started_ns
        pending: list[tuple[int, bytes, tuple[BufferRegion, ...], Any, Any, int]] = []
        for (index, _, key, regions, capacity), lookup in zip(
            valid, lookup_results, strict=True
        ):
            mapping = getattr(lookup, "mapping", None)
            lease = getattr(lookup, "lease", getattr(lookup, "read_lease", None))
            code = _result_code(lookup)
            if (
                mapping is None
                or lease is None
                or code not in {"OK", "SUCCESS", "READY"}
            ):
                results[index] = KVResult(
                    key,
                    self._metadata_failure(code, is_lookup=True),
                    detail=getattr(lookup, "detail", code),
                )
                continue
            logical_length = getattr(mapping, "logical_length")
            if logical_length > capacity:
                results[index] = KVResult(
                    key,
                    KVStatus.RESOURCE_EXHAUSTED,
                    logical_length=logical_length,
                    detail="destination capacity is smaller than stored value",
                )
                self._report_terminal([self._terminal_report(lease)])
                self._release_leases([lease])
                continue
            if len(regions) > 1 and logical_length != capacity:
                results[index] = KVResult(
                    key,
                    KVStatus.INVALID_ARGUMENT,
                    logical_length=logical_length,
                    detail="scatter/gather destination must exactly match stored value",
                )
                self._report_terminal([self._terminal_report(lease)])
                self._release_leases([lease])
                continue
            locations = getattr(mapping, "locations")
            location = locations[0]
            pending.append((index, key, regions, location, lease, logical_length))

        prepare_ns = time.perf_counter_ns() - lookup_ended_ns
        register_ns = 0
        io_build_ns = 0
        submit_ns = 0
        wait_ns = 0
        unregister_ns = 0
        terminal_report_ns = 0
        release_ns = 0
        touch_ns = 0
        if pending:
            flat_regions = [region for entry in pending for region in entry[2]]
            try:
                register_started_ns = time.perf_counter_ns()
                memory = self._agent.register_memory(flat_regions)
                register_ns = time.perf_counter_ns() - register_started_ns
            except BaseException:
                self._abandon_leases([entry[4] for entry in pending])
                raise
            completions = []
            handles: Sequence[Any] = ()
            groups: list[tuple[int, int]] = []
            try:
                io_build_started_ns = time.perf_counter_ns()
                io_items = []
                memory_index = 0
                for index, _, regions, location, lease, logical_length in pending:
                    start = len(io_items)
                    device_offset = location.offset
                    remaining = logical_length
                    for region in regions:
                        length = min(region.length, remaining)
                        if length == 0:
                            break
                        io_items.append(
                            IoItem(
                                location.device_id,
                                device_offset,
                                memory,
                                memory_index,
                                0,
                                length,
                                device_generation=getattr(
                                    location, "device_generation", location.agent_epoch
                                ),
                                authority_epoch=getattr(lease, "service_epoch", None),
                                terminal_report=self._terminal_report_dict(lease),
                                request_id=items[index].request_id,
                            )
                        )
                        memory_index += 1
                        device_offset += length
                        remaining -= length
                    groups.append((start, len(io_items) - start))
                io_build_ns = time.perf_counter_ns() - io_build_started_ns
                submit_started_ns = time.perf_counter_ns()
                handles = self._agent.submit_batch_load(io_items)
                submit_ns = time.perf_counter_ns() - submit_started_ns
                wait_started_ns = time.perf_counter_ns()
                completions = self._safe_wait(handles, started, timeout)
                wait_ns = time.perf_counter_ns() - wait_started_ns
            except BaseException:
                terminal = not handles
                if handles:
                    try:
                        self._agent.wait(handles, 0)
                        terminal = True
                    except BaseException:
                        terminal = False
                tokens = [entry[4] for entry in pending]
                if terminal:
                    self._agent.unregister_memory(memory)
                    self._abandon_leases(tokens)
                else:
                    self._defer_agent_cleanup(handles, memory, tokens, is_write=False)
                raise
            finally:
                if completions:
                    unregister_started_ns = time.perf_counter_ns()
                    self._agent.unregister_memory(memory)
                    unregister_ns = time.perf_counter_ns() - unregister_started_ns

            reports = []
            leases = []
            touches = []
            for (index, key, _, location, lease, logical_length), (
                start,
                count,
            ) in zip(pending, groups, strict=True):
                item_completions = completions[start : start + count]
                failure = next(
                    (
                        completion
                        for completion in item_completions
                        if completion.status is not CompletionStatus.OK
                    ),
                    None,
                )
                success = failure is None
                reports.append(self._terminal_report(lease))
                leases.append(lease)
                if success:
                    touches.append(
                        _make(
                            TouchItem,
                            key=key,
                            client_id=self.client_id,
                            request_id=uuid.uuid4().hex,
                        )
                    )
                    results[index] = KVResult(
                        key,
                        KVStatus.OK,
                        logical_length=logical_length,
                        device_id=getattr(location, "device_id", None),
                    )
                else:
                    status = self._completion_status(failure.status)
                    if status in {
                        KVStatus.UNAVAILABLE,
                        KVStatus.IO_ERROR,
                        KVStatus.DEADLINE_EXCEEDED,
                        KVStatus.CANCELLED,
                    }:
                        status = KVStatus.MISS
                    results[index] = KVResult(key, status, detail=failure.detail)
            terminal_report_started_ns = time.perf_counter_ns()
            self._report_terminal(reports)
            terminal_report_ns = time.perf_counter_ns() - terminal_report_started_ns
            release_started_ns = time.perf_counter_ns()
            self._release_leases(leases)
            release_ns = time.perf_counter_ns() - release_started_ns
            if touches:
                touch_started_ns = time.perf_counter_ns()
                self._metadata("batch_touch", touches)
                touch_ns = time.perf_counter_ns() - touch_started_ns

        final = [result for result in results if result is not None]
        assert len(final) == len(items)
        with self._lock:
            self._metrics["get_items"] += len(items)
            self._metrics["hits"] += sum(
                result.status is KVStatus.OK for result in final
            )
            self._metrics["misses"] += sum(
                result.status is KVStatus.MISS for result in final
            )
            self._metrics["data_bytes"] += sum(
                result.logical_length
                for result in final
                if result.status is KVStatus.OK
            )
        client_ended_ns = time.perf_counter_ns()
        client_e2e_ns = client_ended_ns - started_ns
        emit_trace(
            {
                "event": "client_get",
                "started_ns": started_ns,
                "ended_ns": client_ended_ns,
                "request_ids": [item.request_id for item in items],
                "key_digests": [_key_digest(item.key) for item in items],
                "item_count": len(items),
                "bytes": sum(result.logical_length for result in final if result.ok),
                "statuses": [result.status.value for result in final],
                "lookup_started_ns": lookup_started_ns,
                "lookup_ended_ns": lookup_ended_ns,
                "metadata_lookup_ns": lookup_ns,
                "prepare_ns": prepare_ns,
                "register_memory_ns": register_ns,
                "io_build_ns": io_build_ns,
                "submit_ns": submit_ns,
                "wait_ns": wait_ns,
                "unregister_memory_ns": unregister_ns,
                "terminal_report_ns": terminal_report_ns,
                "release_lease_ns": release_ns,
                "touch_ns": touch_ns,
                "client_e2e_ns": client_e2e_ns,
            }
        )
        return final

    def batch_exists(self, keys: Sequence[bytes | str]) -> list[bool]:
        self._ensure_open()
        if not keys:
            return []
        started_ns = time.perf_counter_ns()
        lookups = [
            _make(
                LookupItem,
                key=_key_bytes(key),
                lease_ttl_s=self._config.read_lease_ttl_s,
                client_id=self.client_id,
                request_id=uuid.uuid4().hex,
            )
            for key in keys
        ]
        lookup_started_ns = time.perf_counter_ns()
        lookup_results = self._metadata("batch_lookup", lookups)
        lookup_ended_ns = time.perf_counter_ns()
        lookup_ns = lookup_ended_ns - lookup_started_ns
        leases = [
            getattr(result, "lease", getattr(result, "read_lease", None))
            for result in lookup_results
        ]
        real_leases = [lease for lease in leases if lease is not None]
        report_started_ns = time.perf_counter_ns()
        self._report_terminal([self._terminal_report(lease) for lease in real_leases])
        report_ns = time.perf_counter_ns() - report_started_ns
        release_started_ns = time.perf_counter_ns()
        self._release_leases(real_leases)
        release_ns = time.perf_counter_ns() - release_started_ns
        found = [
            _result_code(result) in {"OK", "SUCCESS", "READY"}
            and getattr(result, "mapping", None) is not None
            for result in lookup_results
        ]
        with self._lock:
            self._metrics["exists_items"] += len(keys)
        client_ended_ns = time.perf_counter_ns()
        client_e2e_ns = client_ended_ns - started_ns
        emit_trace(
            {
                "event": "client_exists",
                "started_ns": started_ns,
                "ended_ns": client_ended_ns,
                "key_digests": [_key_digest(key) for key in keys],
                "item_count": len(keys),
                "hits": sum(found),
                "lookup_started_ns": lookup_started_ns,
                "lookup_ended_ns": lookup_ended_ns,
                "metadata_lookup_ns": lookup_ns,
                "terminal_report_ns": report_ns,
                "release_lease_ns": release_ns,
                "client_e2e_ns": client_e2e_ns,
            }
        )
        return found

    def submit_batch_set(
        self,
        items_or_keys: Sequence[SetItem] | Sequence[bytes | str],
        sources: Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> list[ClientOperationHandle]:
        items = self._coerce_set_items(items_or_keys, sources)
        return self._submit_async(self.batch_set, items, None, timeout)

    def submit_batch_get(
        self,
        items_or_keys: Sequence[GetItem] | Sequence[bytes | str],
        destinations: Sequence[Any] | None = None,
        timeout: float | None = None,
    ) -> list[ClientOperationHandle]:
        items = self._coerce_get_items(items_or_keys, destinations)
        return self._submit_async(self.batch_get, items, None, timeout)

    def wait(
        self,
        handles: Sequence[ClientOperationHandle],
        timeout: float | None = None,
    ) -> list[KVResult]:
        deadline = None if timeout is None else time.monotonic() + timeout
        results: list[KVResult] = []
        for handle in handles:
            with self._lock:
                record = self._async.get(handle.id)
            if record is None:
                raise KeyError(f"unknown or consumed client operation: {handle.id}")
            batch, index = record
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            values = batch.future.result(timeout=remaining)
            results.append(values[index])
        with self._lock:
            for handle in handles:
                record = self._async.pop(handle.id, None)
                if record is not None:
                    record[0].consumed.add(record[1])
        return results

    def poll(
        self, max_completions: int
    ) -> list[tuple[ClientOperationHandle, KVResult]]:
        if max_completions < 0:
            raise ValueError("max_completions must be non-negative")
        ready: list[ClientOperationHandle] = []
        with self._lock:
            for handle_id, (batch, _) in self._async.items():
                if batch.future.done():
                    ready.append(ClientOperationHandle(handle_id))
                    if len(ready) == max_completions:
                        break
        if not ready:
            return []
        values = self.wait(ready)
        return list(zip(ready, values, strict=True))

    def cancel(self, handles: Sequence[ClientOperationHandle]) -> list[bool]:
        outcomes: list[bool] = []
        with self._lock:
            for handle in handles:
                record = self._async.get(handle.id)
                outcomes.append(False if record is None else record[0].future.cancel())
        return outcomes

    def get_device_id_for_numa_node(self, numa_node: int) -> int | None:
        devices = self._service.list_devices()
        candidates = [
            device.device_id
            for device in devices
            if getattr(device, "numa_node", None) == numa_node
            and bool(getattr(device, "healthy", True))
        ]
        return min(candidates) if candidates else None

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._metrics)

    def close(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._config.default_timeout_s)
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        if self._config.close_agent_on_close:
            self._agent.close(timeout=self._config.default_timeout_s)
        if (
            self._config.heartbeat_device_id is not None
            and self._config.heartbeat_agent_epoch is not None
        ):
            self._service.stop_device(
                self._config.heartbeat_device_id,
                self._config.heartbeat_agent_epoch,
                device_generation=self._config.heartbeat_device_generation,
            )

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._config.heartbeat_interval_s):
            success = False
            try:
                result = self._service.heartbeat(
                    self._config.heartbeat_device_id,
                    self._config.heartbeat_agent_epoch,
                )
                success = _result_code(result) in {"OK", "SUCCESS"}
            except Exception:
                success = False
            with self._lock:
                metric = "heartbeat_successes" if success else "heartbeat_failures"
                self._metrics[metric] += 1

    def _submit_async(
        self, method: Any, items: Sequence[Any], second: Any, timeout: float | None
    ) -> list[ClientOperationHandle]:
        self._ensure_open()
        if not items:
            return []
        if self._async_admission.acquire(blocking=False):
            try:
                future = self._executor.submit(method, items, second, timeout)
            except BaseException:
                self._async_admission.release()
                raise
            future.add_done_callback(lambda _: self._async_admission.release())
        else:
            future = Future()
            future.set_result(
                [
                    KVResult(
                        _key_bytes(item.key),
                        KVStatus.RESOURCE_EXHAUSTED,
                        detail="maximum in-flight client batches reached",
                    )
                    for item in items
                ]
            )
        handles = tuple(ClientOperationHandle(uuid.uuid4().hex) for _ in items)
        batch = _AsyncBatch(future, handles)
        with self._lock:
            for index, handle in enumerate(handles):
                self._async[handle.id] = (batch, index)
        return list(handles)

    def _safe_wait(
        self, handles: Sequence[Any], started: float, timeout: float
    ) -> list[Any]:
        remaining = max(0.0, timeout - (time.monotonic() - started))
        try:
            return self._agent.wait(handles, remaining)
        except OperationTimeoutError:
            # A public deadline never authorizes buffer/extent reuse. Drain the
            # underlying operation before returning a deadline result upstream.
            drained = self._agent.wait(handles, None)
            return [
                type(completion)(
                    completion.handle,
                    CompletionStatus.DEADLINE_EXCEEDED,
                    0,
                    "client deadline expired before terminal completion",
                )
                for completion in drained
            ]

    def _defer_agent_cleanup(
        self,
        handles: Sequence[Any],
        memory: Any,
        tokens: Sequence[Any],
        *,
        is_write: bool,
    ) -> None:
        def cleanup() -> None:
            try:
                self._agent.wait(handles, None)
            except BaseException:
                return
            try:
                self._agent.unregister_memory(memory)
            except BaseException:
                pass
            if is_write:
                self._abandon_reservations(tokens)
            else:
                self._abandon_leases(tokens)

        threading.Thread(
            target=cleanup,
            name="nixl-shard-terminal-cleanup",
            daemon=True,
        ).start()

    def _metadata(self, method_name: str, items: Sequence[Any]) -> list[Any]:
        if not items:
            return []
        max_items = int(getattr(self._service, "max_batch_items", len(items)))
        max_bytes = int(getattr(self._service, "max_batch_bytes", 1 << 20))
        results: list[Any] = []
        method = getattr(self._service, method_name)
        for chunk in _chunks(items, max(1, max_items), max(1, max_bytes)):
            chunk_results = None
            last_error: Exception | None = None
            for _ in range(2):
                with self._lock:
                    self._metrics["metadata_calls"] += 1
                try:
                    chunk_results = method(list(chunk))
                    break
                except Exception as exc:
                    last_error = exc
            if chunk_results is None:
                assert last_error is not None
                raise last_error
            if len(chunk_results) != len(chunk):
                raise RuntimeError(
                    f"{method_name} returned {len(chunk_results)} results "
                    f"for {len(chunk)} items"
                )
            results.extend(chunk_results)
        return results

    def _report_terminal(self, reports: Sequence[Any]) -> None:
        if not reports:
            return
        method = getattr(self._service, "batch_report_io_terminal", None)
        if method is not None:
            results = self._metadata("batch_report_io_terminal", reports)
        else:
            single = getattr(self._service, "report_io_terminal", None)
            if single is None:
                return
            results = []
            for report in reports:
                last_error: Exception | None = None
                for _ in range(2):
                    with self._lock:
                        self._metrics["metadata_calls"] += 1
                    try:
                        results.append(single(report))
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    assert last_error is not None
                    raise last_error
        failures = [
            result
            for result in results
            if _result_code(result) not in {"OK", "SUCCESS"}
        ]
        if failures:
            first = failures[0]
            raise RuntimeError(
                "metadata authority rejected terminal I/O report: "
                f"{_result_code(first)} {getattr(first, 'detail', '')}"
            )

    def _abandon_reservations(self, reservations: Sequence[Any]) -> None:
        reports = [self._terminal_report(item) for item in reservations]
        try:
            self._report_terminal(reports)
        except Exception:
            pass
        aborts = [
            _make(
                AbortItem,
                reservation_id=item.reservation_id,
                generation=item.generation,
                service_epoch=item.service_epoch,
                client_id=self.client_id,
                request_id=uuid.uuid4().hex,
            )
            for item in reservations
        ]
        try:
            self._metadata("batch_abort", aborts)
        except Exception:
            pass

    def _abandon_leases(self, leases: Sequence[Any]) -> None:
        try:
            self._report_terminal([self._terminal_report(item) for item in leases])
        except Exception:
            pass
        try:
            self._release_leases(leases)
        except Exception:
            pass

    def _terminal_report(self, token: Any) -> Any:
        is_write = hasattr(token, "reservation_id")
        return _make(
            IoTerminalReport,
            token_id=(
                getattr(token, "reservation_id")
                if is_write
                else getattr(token, "lease_id")
            ),
            token_kind=IOTokenKind.WRITE if is_write else IOTokenKind.READ,
            device_id=getattr(token, "device_id", None),
            agent_epoch=getattr(token, "agent_epoch", None),
            extent_generation=getattr(
                token, "generation", getattr(token, "extent_generation", None)
            ),
            service_epoch=getattr(token, "service_epoch", None),
            device_generation=getattr(
                token, "device_generation", getattr(token, "agent_epoch", None)
            ),
            client_id=self.client_id,
            request_id=uuid.uuid4().hex,
        )

    def _terminal_report_dict(self, token: Any) -> dict[str, Any]:
        report = self._terminal_report(token)
        return {
            "token_id": report.token_id,
            "token_kind": _enum_name(report.token_kind).lower(),
            "service_epoch": report.service_epoch,
            "device_id": report.device_id,
            "agent_epoch": report.agent_epoch,
            "extent_generation": report.extent_generation,
            "device_generation": getattr(report, "device_generation", None),
            "client_id": report.client_id,
            "request_id": report.request_id,
        }

    def _release_leases(self, leases: Sequence[Any]) -> None:
        if not leases:
            return
        items = [
            _make(
                ReleaseReadItem,
                lease_id=getattr(
                    lease, "lease_id", getattr(lease, "read_lease_id", None)
                ),
                object_generation=getattr(lease, "object_generation", None),
                service_epoch=getattr(lease, "service_epoch", None),
                client_id=self.client_id,
                request_id=uuid.uuid4().hex,
            )
            for lease in leases
        ]
        self._metadata("batch_release_read", items)

    @staticmethod
    def _completion_status(status: CompletionStatus) -> KVStatus:
        try:
            return KVStatus(status.value)
        except ValueError:
            return KVStatus.INTERNAL

    @staticmethod
    def _metadata_failure(code: str, *, is_lookup: bool) -> KVStatus:
        if is_lookup and code in {"", "MISS", "NOT_FOUND", "EVICTED", "STALE_EPOCH"}:
            return KVStatus.MISS
        mapping = {
            "NO_SPACE": KVStatus.RESOURCE_EXHAUSTED,
            "RESOURCE_EXHAUSTED": KVStatus.RESOURCE_EXHAUSTED,
            "INVALID_ARGUMENT": KVStatus.INVALID_ARGUMENT,
            "DEADLINE_EXCEEDED": KVStatus.DEADLINE_EXCEEDED,
            "UNAVAILABLE": KVStatus.UNAVAILABLE,
            "CONFLICT": KVStatus.UNAVAILABLE,
            "RETRYABLE_CONFLICT": KVStatus.UNAVAILABLE,
            "DATA_LOSS": KVStatus.DATA_LOSS,
        }
        return mapping.get(code, KVStatus.INTERNAL)

    @staticmethod
    def _coerce_set_items(
        items_or_keys: Sequence[SetItem] | Sequence[bytes | str],
        sources: Sequence[Any] | None,
    ) -> list[SetItem]:
        values = list(items_or_keys)
        if sources is None:
            if not all(isinstance(item, SetItem) for item in values):
                raise TypeError("sources are required when passing keys")
            return list(values)  # type: ignore[return-value]
        if len(values) != len(sources):
            raise ValueError("keys and sources must have equal length")
        return [
            SetItem(key, source) for key, source in zip(values, sources, strict=True)
        ]

    @staticmethod
    def _coerce_get_items(
        items_or_keys: Sequence[GetItem] | Sequence[bytes | str],
        destinations: Sequence[Any] | None,
    ) -> list[GetItem]:
        values = list(items_or_keys)
        if destinations is None:
            if not all(isinstance(item, GetItem) for item in values):
                raise TypeError("destinations are required when passing keys")
            return list(values)  # type: ignore[return-value]
        if len(values) != len(destinations):
            raise ValueError("keys and destinations must have equal length")
        return [
            GetItem(key, destination)
            for key, destination in zip(values, destinations, strict=True)
        ]

    def _ensure_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("client is closed")

    def __enter__(self) -> ShardClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
