# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory metadata authority and extent allocator for NIXLShard.

The service is deliberately independent of its transport.  It is safe to use
directly in tests and embedded deployments; :mod:`nixl_shard.service_rpc`
provides the small loopback TCP transport used by the functional tests.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Union


class ResultCode(str, Enum):
    OK = "ok"
    EXISTING_READY = "existing_ready"
    RETRYABLE_CONFLICT = "retryable_conflict"
    NO_SPACE = "no_space"
    UNAVAILABLE = "unavailable"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    STALE_TOKEN = "stale_token"
    CONFLICT = "conflict"
    INTERNAL = "internal"


class IOTokenKind(str, Enum):
    WRITE = "write"
    READ = "read"


class DeviceState(str, Enum):
    ONLINE = "online"
    DRAINING = "draining"
    OFFLINE = "offline"


class BatchRejectedError(ValueError):
    """The whole batch was rejected before any item was executed."""


@dataclass(frozen=True)
class ServiceConfig:
    max_batch_items: int = 256
    max_batch_bytes: int = 1 << 20
    default_reservation_ttl_s: float = 30.0
    default_read_lease_ttl_s: float = 30.0
    device_timeout_s: float = 30.0
    low_watermark: float = 0.60
    high_watermark: float = 0.80
    auto_evict: bool = True
    clock: Callable[[], float] = time.monotonic
    device_stop_grace_s: float = 30.0
    idempotency_ttl_s: float = 300.0
    max_idempotency_entries: int = 100_000

    def __post_init__(self) -> None:
        if self.max_batch_items <= 0 or self.max_batch_bytes <= 0:
            raise ValueError("batch limits must be positive")
        if self.default_reservation_ttl_s <= 0 or self.default_read_lease_ttl_s <= 0:
            raise ValueError("TTLs must be positive")
        if self.device_timeout_s <= 0:
            raise ValueError("device timeout must be positive")
        if self.device_stop_grace_s < 0:
            raise ValueError("device stop grace must be non-negative")
        if self.idempotency_ttl_s <= 0 or self.max_idempotency_entries <= 0:
            raise ValueError("idempotency retention limits must be positive")
        _validate_watermarks(self.low_watermark, self.high_watermark)


@dataclass(frozen=True)
class DeviceRegistration:
    device_id: int
    agent_endpoint: str
    capacity_bytes: int
    allocation_alignment: int
    agent_epoch: int
    numa_node: int = -1
    failure_domain: Optional[str] = None
    device_generation: int = 1


@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    agent_endpoint: str
    capacity_bytes: int
    allocation_alignment: int
    agent_epoch: int
    numa_node: int
    failure_domain: Optional[str]
    healthy: bool
    last_heartbeat: float
    device_generation: int = 1
    state: DeviceState = DeviceState.ONLINE
    drain_started_at: Optional[float] = None
    offline_at: Optional[float] = None
    outstanding_reservations: int = 0
    outstanding_read_leases: int = 0
    outstanding_io: int = 0


@dataclass(frozen=True)
class OperationResult:
    code: ResultCode
    detail: str = ""


@dataclass(frozen=True)
class DeviceRegistrationResult(OperationResult):
    device: Optional[DeviceInfo] = None


@dataclass(frozen=True)
class DeviceManagementResult(OperationResult):
    device: Optional[DeviceInfo] = None


@dataclass(frozen=True)
class ReserveItem:
    key: bytes
    logical_length: int
    client_id: str
    request_id: str
    preferred_device_id: Optional[int] = None
    reservation_ttl_s: Optional[float] = None
    deadline: Optional[float] = None


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    key_digest: str
    device_id: int
    agent_endpoint: str
    agent_epoch: int
    offset: int
    allocated_length: int
    logical_length: int
    generation: int
    service_epoch: int
    expires_at: float
    device_generation: int = 1


@dataclass(frozen=True)
class Location:
    device_id: int
    agent_endpoint: str
    agent_epoch: int
    offset: int
    allocated_length: int
    extent_generation: int
    device_generation: int = 1


@dataclass(frozen=True)
class ObjectMapping:
    key_digest: str
    logical_length: int
    locations: tuple[Location, ...]
    object_generation: int
    service_epoch: int


@dataclass(frozen=True)
class ReserveResult(OperationResult):
    reservation: Optional[Reservation] = None
    mapping: Optional[ObjectMapping] = None
    retry_after_ms: Optional[int] = None


@dataclass(frozen=True)
class CommitItem:
    reservation_id: str
    generation: int
    service_epoch: int
    client_id: str
    request_id: str
    deadline: Optional[float] = None


@dataclass(frozen=True)
class CommitResult(OperationResult):
    mapping: Optional[ObjectMapping] = None


@dataclass(frozen=True)
class AbortItem:
    reservation_id: str
    generation: int
    service_epoch: int
    client_id: str
    request_id: str
    deadline: Optional[float] = None


@dataclass(frozen=True)
class LookupItem:
    key: bytes
    client_id: str
    request_id: str
    lease_ttl_s: Optional[float] = None
    preferred_device_id: Optional[int] = None
    deadline: Optional[float] = None


@dataclass(frozen=True)
class ReadLease:
    lease_id: str
    key_digest: str
    device_id: int
    agent_epoch: int
    extent_generation: int
    object_generation: int
    service_epoch: int
    expires_at: float
    device_generation: int = 1


@dataclass(frozen=True)
class LookupResult(OperationResult):
    mapping: Optional[ObjectMapping] = None
    lease: Optional[ReadLease] = None


@dataclass(frozen=True)
class RenewReadItem:
    lease_id: str
    object_generation: int
    service_epoch: int
    client_id: str
    request_id: str
    lease_ttl_s: Optional[float] = None
    deadline: Optional[float] = None


@dataclass(frozen=True)
class RenewReadResult(OperationResult):
    lease: Optional[ReadLease] = None


@dataclass(frozen=True)
class ReleaseReadItem:
    lease_id: str
    object_generation: int
    service_epoch: int
    client_id: str
    request_id: str
    deadline: Optional[float] = None


@dataclass(frozen=True)
class TouchItem:
    key: bytes
    client_id: str
    request_id: str
    deadline: Optional[float] = None


@dataclass(frozen=True)
class IoTerminalReport:
    token_id: str
    token_kind: IOTokenKind
    service_epoch: int
    device_id: int
    agent_epoch: int
    extent_generation: int
    client_id: str
    request_id: str
    deadline: Optional[float] = None
    device_generation: Optional[int] = None


@dataclass(frozen=True)
class CapacityReport:
    device_id: int
    capacity_bytes: int
    free_bytes: int
    reserved_bytes: int
    ready_bytes: int
    reclaim_pending_bytes: int
    largest_free_extent: int
    low_watermark: float
    high_watermark: float


@dataclass(frozen=True)
class EvictionResult(OperationResult):
    selected_objects: int = 0
    selected_bytes: int = 0
    reclaimed_bytes: int = 0


@dataclass
class _Device:
    registration: DeviceRegistration
    free: list[tuple[int, int]]
    healthy: bool
    last_heartbeat: float
    fenced: bool = False
    next_extent_generation: int = 1
    low_watermark: float = 0.60
    high_watermark: float = 0.80
    state: DeviceState = DeviceState.ONLINE
    drain_started_at: Optional[float] = None
    drain_deadline: Optional[float] = None
    offline_at: Optional[float] = None


@dataclass
class _ReservationRecord:
    public: Reservation
    key: bytes
    state: str = "reserved"
    write_terminal: bool = False


@dataclass
class _ObjectRecord:
    key: bytes
    mapping: ObjectMapping
    state: str
    last_access: float
    write_terminal: bool = True
    lease_ids: set[str] = field(default_factory=set)


@dataclass
class _LeaseRecord:
    public: ReadLease
    object_generation: int
    active: bool = True
    io_terminal: bool = False


@dataclass(frozen=True)
class _CachedResult:
    fingerprint: str
    result: object
    created_at: float


def _validate_watermarks(low: float, high: float) -> None:
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("watermarks must satisfy 0 <= low < high <= 1")


def _aligned_size(length: int, alignment: int) -> int:
    return (length + alignment - 1) // alignment * alignment


class ShardNamingService:
    """Thread-safe, single-authority V1 naming and allocation service."""

    def __init__(
        self, config: Optional[ServiceConfig] = None, *, epoch: Optional[int] = None
    ):
        self.config = config or ServiceConfig()
        if epoch is None:
            # A fresh process must not accept tokens from an earlier service
            # incarnation. Supervisors may instead supply a persisted epoch.
            epoch = (uuid.uuid4().int & ((1 << 63) - 1)) or 1
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        self._epoch = epoch
        self._lock = threading.RLock()
        self._devices: dict[int, _Device] = {}
        self._minimum_agent_epoch: dict[int, int] = {}
        self._minimum_device_generation: dict[int, int] = {}
        self._reservations: dict[str, _ReservationRecord] = {}
        self._reservation_by_key: dict[bytes, str] = {}
        self._objects_by_key: dict[bytes, _ObjectRecord] = {}
        self._objects_by_generation: dict[int, _ObjectRecord] = {}
        self._leases: dict[str, _LeaseRecord] = {}
        self._idempotency: dict[tuple[str, str, str], _CachedResult] = {}
        self._next_object_generation = 1
        self._metrics: dict[str, int] = {
            "reservations": 0,
            "commits": 0,
            "aborts": 0,
            "lookups": 0,
            "lookup_misses": 0,
            "evictions": 0,
            "idempotency_hits": 0,
            "idempotency_evictions": 0,
            "reclaimed_extents": 0,
            "device_stops": 0,
            "device_forced_offlines": 0,
        }

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def max_batch_items(self) -> int:
        return self.config.max_batch_items

    @property
    def max_batch_bytes(self) -> int:
        return self.config.max_batch_bytes

    def register_device(
        self, registration: DeviceRegistration
    ) -> DeviceRegistrationResult:
        with self._lock:
            try:
                self._validate_registration(registration)
            except ValueError as exc:
                return DeviceRegistrationResult(ResultCode.INVALID_ARGUMENT, str(exc))
            self._maintenance(reap_only=True)
            minimum_epoch = self._minimum_agent_epoch.get(registration.device_id)
            if minimum_epoch is not None and registration.agent_epoch < minimum_epoch:
                return DeviceRegistrationResult(
                    ResultCode.STALE_TOKEN,
                    f"agent epoch must be at least {minimum_epoch} after fencing",
                )
            minimum_generation = self._minimum_device_generation.get(
                registration.device_id
            )
            if (
                minimum_generation is not None
                and registration.device_generation < minimum_generation
            ):
                return DeviceRegistrationResult(
                    ResultCode.STALE_TOKEN,
                    "device generation must be at least "
                    f"{minimum_generation} after replacement",
                )
            existing = self._devices.get(registration.device_id)
            if existing is not None:
                if (
                    existing.registration == registration
                    and not existing.fenced
                    and existing.state is DeviceState.ONLINE
                ):
                    existing.healthy = True
                    existing.last_heartbeat = self.config.clock()
                    return DeviceRegistrationResult(
                        ResultCode.OK, device=self._device_info(existing)
                    )
                if existing.state is not DeviceState.OFFLINE:
                    return DeviceRegistrationResult(
                        ResultCode.CONFLICT,
                        "device ID already has a live or draining registration",
                        self._device_info(existing),
                    )
                if (
                    registration.agent_epoch <= existing.registration.agent_epoch
                    or registration.device_generation
                    <= existing.registration.device_generation
                ):
                    return DeviceRegistrationResult(
                        ResultCode.STALE_TOKEN,
                        "replacement requires strictly newer agent epoch and "
                        "device generation",
                        self._device_info(existing),
                    )
            now = self.config.clock()
            device = _Device(
                registration,
                [(0, registration.capacity_bytes)],
                True,
                now,
                low_watermark=self.config.low_watermark,
                high_watermark=self.config.high_watermark,
            )
            self._devices[registration.device_id] = device
            self._minimum_agent_epoch.pop(registration.device_id, None)
            self._minimum_device_generation.pop(registration.device_id, None)
            return DeviceRegistrationResult(
                ResultCode.OK, device=self._device_info(device)
            )

    def heartbeat(self, device_id: int, agent_epoch: int) -> OperationResult:
        with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                return OperationResult(ResultCode.NOT_FOUND, "unknown device")
            if device.registration.agent_epoch != agent_epoch or device.fenced:
                return OperationResult(ResultCode.STALE_TOKEN, "stale agent epoch")
            if device.state is not DeviceState.ONLINE:
                return OperationResult(ResultCode.UNAVAILABLE, "device is not online")
            device.healthy = True
            device.last_heartbeat = self.config.clock()
            return OperationResult(ResultCode.OK)

    def list_devices(self) -> tuple[DeviceInfo, ...]:
        with self._lock:
            self._maintenance(reap_only=True)
            return tuple(self._device_info(d) for _, d in sorted(self._devices.items()))

    def get_device_status(self, device_id: int) -> DeviceManagementResult:
        with self._lock:
            self._maintenance(reap_only=True)
            device = self._devices.get(device_id)
            if device is None:
                return DeviceManagementResult(ResultCode.NOT_FOUND, "unknown device")
            return DeviceManagementResult(
                ResultCode.OK, device=self._device_info(device)
            )

    def stop_device(
        self,
        device_id: int,
        agent_epoch: int,
        *,
        device_generation: Optional[int] = None,
        grace_period_s: Optional[float] = None,
    ) -> DeviceManagementResult:
        """Stop new placement/lookups and drain outstanding device operations."""

        with self._lock:
            self._maintenance(reap_only=True)
            device = self._devices.get(device_id)
            if device is None:
                return DeviceManagementResult(ResultCode.NOT_FOUND, "unknown device")
            if device.registration.agent_epoch != agent_epoch:
                return DeviceManagementResult(
                    ResultCode.STALE_TOKEN, "stale agent epoch"
                )
            if (
                device_generation is not None
                and device.registration.device_generation != device_generation
            ):
                return DeviceManagementResult(
                    ResultCode.STALE_TOKEN, "stale device generation"
                )
            if grace_period_s is None:
                grace_period_s = self.config.device_stop_grace_s
            if grace_period_s < 0:
                return DeviceManagementResult(
                    ResultCode.INVALID_ARGUMENT,
                    "device stop grace must be non-negative",
                )
            if device.state is DeviceState.ONLINE:
                self._begin_device_stop(device, grace_period_s)
            self._advance_device_lifecycle(self.config.clock())
            return DeviceManagementResult(
                ResultCode.OK, device=self._device_info(device)
            )

    def drain_device(
        self,
        device_id: int,
        agent_epoch: int,
        *,
        device_generation: Optional[int] = None,
        grace_period_s: Optional[float] = None,
    ) -> DeviceManagementResult:
        return self.stop_device(
            device_id,
            agent_epoch,
            device_generation=device_generation,
            grace_period_s=grace_period_s,
        )

    def batch_reserve(
        self,
        items: Sequence[ReserveItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[ReserveResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            return [
                self._idempotent(
                    "reserve",
                    item,
                    lambda item=item: self._reserve(item),
                    lambda code, detail: ReserveResult(code, detail),
                )
                for item in items
            ]

    def batch_commit(
        self,
        items: Sequence[CommitItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[CommitResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            return [
                self._idempotent(
                    "commit",
                    item,
                    lambda item=item: self._commit(item),
                    lambda code, detail: CommitResult(code, detail),
                )
                for item in items
            ]

    def batch_abort(
        self,
        items: Sequence[AbortItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[OperationResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            return [
                self._idempotent(
                    "abort",
                    item,
                    lambda item=item: self._abort(item),
                    OperationResult,
                )
                for item in items
            ]

    def batch_lookup(
        self,
        items: Sequence[LookupItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[LookupResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            return [
                self._idempotent(
                    "lookup",
                    item,
                    lambda item=item: self._lookup(item),
                    lambda code, detail: LookupResult(code, detail),
                )
                for item in items
            ]

    def batch_renew_read(
        self,
        items: Sequence[RenewReadItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[RenewReadResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            return [
                self._idempotent(
                    "renew_read",
                    item,
                    lambda item=item: self._renew_read(item),
                    lambda code, detail: RenewReadResult(code, detail),
                )
                for item in items
            ]

    def batch_release_read(
        self,
        items: Sequence[ReleaseReadItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[OperationResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            results = [
                self._idempotent(
                    "release_read",
                    item,
                    lambda item=item: self._release_read(item),
                    OperationResult,
                )
                for item in items
            ]
            self._advance_device_lifecycle(self.config.clock())
            return results

    def batch_touch(
        self,
        items: Sequence[TouchItem],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[OperationResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            return [
                self._idempotent(
                    "touch",
                    item,
                    lambda item=item: self._touch(item),
                    OperationResult,
                )
                for item in items
            ]

    def batch_report_io_terminal(
        self,
        items: Sequence[IoTerminalReport],
        *,
        batch_request_id: str = "",
        deadline: Optional[float] = None,
    ) -> list[OperationResult]:
        with self._lock:
            self._preflight(items, deadline, batch_request_id)
            self._maintenance()
            results = [
                self._idempotent(
                    "io_terminal",
                    item,
                    lambda item=item: self._report_io_terminal(item),
                    OperationResult,
                )
                for item in items
            ]
            self._advance_device_lifecycle(self.config.clock())
            return results

    def report_io_terminal(self, item: IoTerminalReport) -> OperationResult:
        return self.batch_report_io_terminal([item])[0]

    def set_watermarks(
        self, device_id: int, low: float, high: float
    ) -> OperationResult:
        with self._lock:
            try:
                _validate_watermarks(low, high)
            except ValueError as exc:
                return OperationResult(ResultCode.INVALID_ARGUMENT, str(exc))
            device = self._devices.get(device_id)
            if device is None:
                return OperationResult(ResultCode.NOT_FOUND, "unknown device")
            device.low_watermark = low
            device.high_watermark = high
            return OperationResult(ResultCode.OK)

    def run_eviction(
        self, device_id: Optional[int] = None, *, force: bool = False
    ) -> EvictionResult:
        with self._lock:
            self._maintenance(reap_only=True)
            ids = [device_id] if device_id is not None else sorted(self._devices)
            selected_objects = selected_bytes = reclaimed_bytes = 0
            for current_id in ids:
                device = self._devices.get(current_id)
                if device is None:
                    if device_id is not None:
                        return EvictionResult(ResultCode.NOT_FOUND, "unknown device")
                    continue
                capacity_before = self._capacity(device)
                free_before = capacity_before.free_bytes
                logical_used = (
                    capacity_before.reserved_bytes + capacity_before.ready_bytes
                )
                if (
                    not force
                    and logical_used
                    <= device.high_watermark * device.registration.capacity_bytes
                ):
                    continue
                target = math.floor(
                    device.low_watermark * device.registration.capacity_bytes
                )
                candidates = sorted(
                    (
                        obj
                        for obj in self._objects_by_generation.values()
                        if obj.state == "ready"
                        and obj.mapping.locations[0].device_id == current_id
                    ),
                    key=lambda obj: (obj.last_access, obj.mapping.object_generation),
                )
                for obj in candidates:
                    if logical_used <= target:
                        break
                    length = obj.mapping.locations[0].allocated_length
                    self._mark_object_reclaim_pending(obj)
                    selected_objects += 1
                    selected_bytes += length
                    logical_used -= length
                reclaimed_bytes += max(
                    0, self._capacity(device).free_bytes - free_before
                )
            if selected_objects:
                self._metrics["evictions"] += selected_objects
            detail = ""
            if not selected_objects:
                detail = "below high watermark or no READY eviction candidates"
            elif reclaimed_bytes < selected_bytes:
                detail = "selected extents await lease release or terminal I/O"
            return EvictionResult(
                ResultCode.OK,
                detail,
                selected_objects,
                selected_bytes,
                reclaimed_bytes,
            )

    def get_capacity(
        self, device_id: Optional[int] = None
    ) -> Union[CapacityReport, tuple[CapacityReport, ...]]:
        with self._lock:
            self._maintenance(reap_only=True)
            if device_id is not None:
                device = self._devices.get(device_id)
                if device is None:
                    raise KeyError(f"unknown device {device_id}")
                return self._capacity(device)
            return tuple(
                self._capacity(self._devices[i]) for i in sorted(self._devices)
            )

    def reap_expired(self) -> OperationResult:
        with self._lock:
            self._maintenance(reap_only=True)
            return OperationResult(ResultCode.OK)

    def fence_device_epoch(
        self,
        device_id: int,
        agent_epoch: int,
        *,
        quiesced: bool,
        device_generation: Optional[int] = None,
    ) -> OperationResult:
        with self._lock:
            self._maintenance(reap_only=True)
            device = self._devices.get(device_id)
            if device is None:
                return OperationResult(ResultCode.NOT_FOUND, "unknown device")
            if device.registration.agent_epoch != agent_epoch:
                return OperationResult(ResultCode.STALE_TOKEN, "stale agent epoch")
            if (
                device_generation is not None
                and device.registration.device_generation != device_generation
            ):
                return OperationResult(
                    ResultCode.STALE_TOKEN, "stale device generation"
                )
            if device.state is DeviceState.ONLINE:
                self._begin_device_stop(device, self.config.device_stop_grace_s)
            device.fenced = True
            if not quiesced:
                return OperationResult(
                    ResultCode.OK, "device fenced; quiescence pending"
                )
            self._transition_device_offline(device, self.config.clock(), forced=True)
            return OperationResult(ResultCode.OK, "device is offline")

    def restart(self) -> int:
        """Discard cache metadata and advance the authority epoch.

        Devices must re-register.  A real deployment calls this only after the
        preceding service process/agent epoch has been fenced and quiesced.
        """
        with self._lock:
            self._epoch += 1
            for device_id, device in self._devices.items():
                self._minimum_agent_epoch[device_id] = max(
                    self._minimum_agent_epoch.get(device_id, 1),
                    device.registration.agent_epoch + 1,
                )
                self._minimum_device_generation[device_id] = max(
                    self._minimum_device_generation.get(device_id, 1),
                    device.registration.device_generation + 1,
                )
            self._devices.clear()
            self._reservations.clear()
            self._reservation_by_key.clear()
            self._objects_by_key.clear()
            self._objects_by_generation.clear()
            self._leases.clear()
            self._idempotency.clear()
            self._next_object_generation = 1
            return self._epoch

    def health(self) -> dict[str, object]:
        with self._lock:
            self._maintenance(reap_only=True)
            return {
                "ok": True,
                "service_epoch": self._epoch,
                "devices": len(self._devices),
                "healthy_devices": sum(d.healthy for d in self._devices.values()),
                "online_devices": sum(
                    d.state is DeviceState.ONLINE for d in self._devices.values()
                ),
                "draining_devices": sum(
                    d.state is DeviceState.DRAINING for d in self._devices.values()
                ),
                "offline_devices": sum(
                    d.state is DeviceState.OFFLINE for d in self._devices.values()
                ),
            }

    def get_object_debug(self, key_or_digest: Union[bytes, str]) -> dict[str, object]:
        with self._lock:
            digest = (
                self._key_digest(key_or_digest)
                if isinstance(key_or_digest, bytes)
                else key_or_digest
            )
            for obj in self._objects_by_generation.values():
                if obj.mapping.key_digest == digest:
                    return {
                        "key_digest": digest,
                        "state": obj.state,
                        "mapping": obj.mapping,
                        "last_access": obj.last_access,
                        "leases": tuple(sorted(obj.lease_ids)),
                    }
            return {"key_digest": digest, "state": "not_found"}

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            metrics = dict(self._metrics)
            metrics.update(
                {
                    "devices": len(self._devices),
                    "reservations_active": sum(
                        r.state == "reserved" for r in self._reservations.values()
                    ),
                    "objects_ready": sum(
                        o.state == "ready" for o in self._objects_by_generation.values()
                    ),
                    "leases_active": sum(l.active for l in self._leases.values()),
                    "idempotency_entries": len(self._idempotency),
                }
            )
            return metrics

    def validate_invariants(self) -> None:
        """Raise AssertionError if allocator/accounting invariants are broken."""
        with self._lock:
            for device_id, device in self._devices.items():
                intervals: list[tuple[int, int, str]] = [
                    (start, start + length, "free") for start, length in device.free
                ]
                for reservation in self._reservations.values():
                    if reservation.public.device_id == device_id:
                        intervals.append(
                            (
                                reservation.public.offset,
                                reservation.public.offset
                                + reservation.public.allocated_length,
                                reservation.state,
                            )
                        )
                for obj in self._objects_by_generation.values():
                    location = obj.mapping.locations[0]
                    if location.device_id == device_id:
                        intervals.append(
                            (
                                location.offset,
                                location.offset + location.allocated_length,
                                obj.state,
                            )
                        )
                intervals.sort()
                assert intervals and intervals[0][0] == 0
                cursor = 0
                for start, end, _ in intervals:
                    assert start == cursor, (device_id, intervals)
                    assert start < end <= device.registration.capacity_bytes
                    cursor = end
                assert cursor == device.registration.capacity_bytes
                report = self._capacity(device)
                assert (
                    report.free_bytes
                    + report.reserved_bytes
                    + report.ready_bytes
                    + report.reclaim_pending_bytes
                    == report.capacity_bytes
                )

    def _reserve(self, item: ReserveItem) -> ReserveResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return ReserveResult(*error)
        if not isinstance(item.key, bytes) or not item.key:
            return ReserveResult(
                ResultCode.INVALID_ARGUMENT, "key must be non-empty bytes"
            )
        if item.logical_length <= 0:
            return ReserveResult(
                ResultCode.INVALID_ARGUMENT, "logical length must be positive"
            )
        ttl = item.reservation_ttl_s or self.config.default_reservation_ttl_s
        if ttl <= 0:
            return ReserveResult(
                ResultCode.INVALID_ARGUMENT, "reservation TTL must be positive"
            )
        ready = self._objects_by_key.get(item.key)
        if ready is not None and ready.state == "ready":
            return ReserveResult(ResultCode.EXISTING_READY, mapping=ready.mapping)
        active_id = self._reservation_by_key.get(item.key)
        if active_id is not None:
            active = self._reservations.get(active_id)
            if active is not None and active.state == "reserved":
                return ReserveResult(
                    ResultCode.RETRYABLE_CONFLICT,
                    "another reservation owns the key",
                    retry_after_ms=min(
                        1000,
                        max(
                            1,
                            int(
                                (active.public.expires_at - self.config.clock()) * 1000
                            ),
                        ),
                    ),
                )
        devices = self._placement_order(item.preferred_device_id)
        if not devices:
            return ReserveResult(ResultCode.UNAVAILABLE, "no healthy devices")
        chosen: Optional[tuple[_Device, int, int]] = None
        for device in devices:
            allocated = _aligned_size(
                item.logical_length, device.registration.allocation_alignment
            )
            offset = self._allocate(device, allocated)
            if offset is None and self.config.auto_evict:
                self.run_eviction(device.registration.device_id, force=True)
                offset = self._allocate(device, allocated)
            if offset is not None:
                chosen = (device, offset, allocated)
                break
        if chosen is None:
            return ReserveResult(ResultCode.NO_SPACE, "no aligned extent fits")
        device, offset, allocated = chosen
        generation = device.next_extent_generation
        device.next_extent_generation += 1
        now = self.config.clock()
        public = Reservation(
            str(uuid.uuid4()),
            self._key_digest(item.key),
            device.registration.device_id,
            device.registration.agent_endpoint,
            device.registration.agent_epoch,
            offset,
            allocated,
            item.logical_length,
            generation,
            self._epoch,
            now + ttl,
            device.registration.device_generation,
        )
        record = _ReservationRecord(public, item.key)
        self._reservations[public.reservation_id] = record
        self._reservation_by_key[item.key] = public.reservation_id
        self._metrics["reservations"] += 1
        if self.config.auto_evict:
            capacity = self._capacity(device)
            logical = capacity.reserved_bytes + capacity.ready_bytes
            if logical > device.high_watermark * device.registration.capacity_bytes:
                self.run_eviction(device.registration.device_id)
        return ReserveResult(ResultCode.OK, reservation=public)

    def _commit(self, item: CommitItem) -> CommitResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return CommitResult(*error)
        if item.service_epoch != self._epoch:
            return CommitResult(ResultCode.STALE_TOKEN, "stale service epoch")
        record = self._reservations.get(item.reservation_id)
        if record is None:
            return CommitResult(ResultCode.NOT_FOUND, "unknown reservation")
        stale = self._validate_reservation_token(
            record, item.service_epoch, item.generation
        )
        if stale is not None:
            return CommitResult(stale, "stale reservation token")
        if record.state != "reserved":
            return CommitResult(
                ResultCode.STALE_TOKEN, "reservation is no longer active"
            )
        if record.public.expires_at <= self.config.clock():
            self._mark_reservation_reclaim_pending(record)
            self._try_reclaim_reservation(record)
            return CommitResult(ResultCode.STALE_TOKEN, "reservation expired")
        if record.key in self._objects_by_key:
            return CommitResult(ResultCode.CONFLICT, "key was concurrently published")
        device = self._devices.get(record.public.device_id)
        if (
            device is None
            or not device.healthy
            or device.state is not DeviceState.ONLINE
            or device.registration.agent_epoch != record.public.agent_epoch
            or device.registration.device_generation != record.public.device_generation
        ):
            self._mark_reservation_reclaim_pending(record)
            return CommitResult(ResultCode.UNAVAILABLE, "device epoch is unavailable")
        location = Location(
            record.public.device_id,
            record.public.agent_endpoint,
            record.public.agent_epoch,
            record.public.offset,
            record.public.allocated_length,
            record.public.generation,
            record.public.device_generation,
        )
        object_generation = self._next_object_generation
        self._next_object_generation += 1
        mapping = ObjectMapping(
            record.public.key_digest,
            record.public.logical_length,
            (location,),
            object_generation,
            self._epoch,
        )
        obj = _ObjectRecord(record.key, mapping, "ready", self.config.clock())
        self._objects_by_key[record.key] = obj
        self._objects_by_generation[object_generation] = obj
        record.state = "committed"
        record.write_terminal = True  # Commit asserts the preceding write completed.
        self._reservation_by_key.pop(record.key, None)
        del self._reservations[item.reservation_id]
        self._metrics["commits"] += 1
        return CommitResult(ResultCode.OK, mapping=mapping)

    def _abort(self, item: AbortItem) -> OperationResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return OperationResult(*error)
        if item.service_epoch != self._epoch:
            return OperationResult(ResultCode.STALE_TOKEN, "stale service epoch")
        record = self._reservations.get(item.reservation_id)
        if record is None:
            return OperationResult(ResultCode.NOT_FOUND, "unknown reservation")
        stale = self._validate_reservation_token(
            record, item.service_epoch, item.generation
        )
        if stale is not None:
            return OperationResult(stale, "stale reservation token")
        if record.state != "reserved":
            return OperationResult(
                ResultCode.STALE_TOKEN, "reservation is no longer active"
            )
        self._mark_reservation_reclaim_pending(record)
        self._try_reclaim_reservation(record)
        self._metrics["aborts"] += 1
        return OperationResult(ResultCode.OK)

    def _lookup(self, item: LookupItem) -> LookupResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return LookupResult(*error)
        if not isinstance(item.key, bytes) or not item.key:
            return LookupResult(
                ResultCode.INVALID_ARGUMENT, "key must be non-empty bytes"
            )
        ttl = item.lease_ttl_s or self.config.default_read_lease_ttl_s
        if ttl <= 0:
            return LookupResult(
                ResultCode.INVALID_ARGUMENT, "lease TTL must be positive"
            )
        obj = self._objects_by_key.get(item.key)
        if obj is None or obj.state != "ready":
            self._metrics["lookup_misses"] += 1
            return LookupResult(ResultCode.NOT_FOUND, "cache miss")
        location = obj.mapping.locations[0]
        device = self._devices.get(location.device_id)
        if (
            device is None
            or not device.healthy
            or device.state is not DeviceState.ONLINE
            or device.registration.agent_epoch != location.agent_epoch
            or device.registration.device_generation != location.device_generation
        ):
            return LookupResult(ResultCode.UNAVAILABLE, "object device is unavailable")
        lease = ReadLease(
            str(uuid.uuid4()),
            obj.mapping.key_digest,
            location.device_id,
            location.agent_epoch,
            location.extent_generation,
            obj.mapping.object_generation,
            self._epoch,
            self.config.clock() + ttl,
            location.device_generation,
        )
        self._leases[lease.lease_id] = _LeaseRecord(
            lease, obj.mapping.object_generation
        )
        obj.lease_ids.add(lease.lease_id)
        obj.last_access = self.config.clock()
        self._metrics["lookups"] += 1
        return LookupResult(ResultCode.OK, mapping=obj.mapping, lease=lease)

    def _renew_read(self, item: RenewReadItem) -> RenewReadResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return RenewReadResult(*error)
        if item.service_epoch != self._epoch:
            return RenewReadResult(ResultCode.STALE_TOKEN, "stale service epoch")
        record = self._leases.get(item.lease_id)
        if record is None:
            return RenewReadResult(ResultCode.NOT_FOUND, "unknown lease")
        if (
            item.service_epoch != self._epoch
            or item.object_generation != record.object_generation
        ):
            return RenewReadResult(ResultCode.STALE_TOKEN, "stale lease token")
        if not record.active or record.public.expires_at <= self.config.clock():
            record.active = False
            return RenewReadResult(ResultCode.STALE_TOKEN, "lease expired or released")
        device = self._devices.get(record.public.device_id)
        if (
            device is None
            or device.state is not DeviceState.ONLINE
            or device.registration.agent_epoch != record.public.agent_epoch
            or device.registration.device_generation != record.public.device_generation
        ):
            return RenewReadResult(ResultCode.UNAVAILABLE, "device is not online")
        ttl = item.lease_ttl_s or self.config.default_read_lease_ttl_s
        if ttl <= 0:
            return RenewReadResult(
                ResultCode.INVALID_ARGUMENT, "lease TTL must be positive"
            )
        record.public = ReadLease(
            record.public.lease_id,
            record.public.key_digest,
            record.public.device_id,
            record.public.agent_epoch,
            record.public.extent_generation,
            record.public.object_generation,
            record.public.service_epoch,
            self.config.clock() + ttl,
            record.public.device_generation,
        )
        return RenewReadResult(ResultCode.OK, lease=record.public)

    def _release_read(self, item: ReleaseReadItem) -> OperationResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return OperationResult(*error)
        if item.service_epoch != self._epoch:
            return OperationResult(ResultCode.STALE_TOKEN, "stale service epoch")
        record = self._leases.get(item.lease_id)
        if record is None:
            return OperationResult(ResultCode.NOT_FOUND, "unknown lease")
        if (
            item.service_epoch != self._epoch
            or item.object_generation != record.object_generation
        ):
            return OperationResult(ResultCode.STALE_TOKEN, "stale lease token")
        record.active = False
        self._cleanup_lease_if_safe(record)
        return OperationResult(ResultCode.OK)

    def _touch(self, item: TouchItem) -> OperationResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return OperationResult(*error)
        obj = self._objects_by_key.get(item.key)
        if obj is None or obj.state != "ready":
            return OperationResult(ResultCode.NOT_FOUND, "cache miss")
        obj.last_access = self.config.clock()
        return OperationResult(ResultCode.OK)

    def _report_io_terminal(self, item: IoTerminalReport) -> OperationResult:
        error = self._validate_identity_and_deadline(item)
        if error is not None:
            return OperationResult(*error)
        if item.service_epoch != self._epoch:
            return OperationResult(ResultCode.STALE_TOKEN, "stale service epoch")
        device = self._devices.get(item.device_id)
        minimum_epoch = self._minimum_agent_epoch.get(item.device_id)
        minimum_generation = self._minimum_device_generation.get(item.device_id)
        if (
            (device is not None and device.state is DeviceState.OFFLINE)
            or (
                device is not None
                and device.registration.agent_epoch != item.agent_epoch
            )
            or (
                device is not None
                and item.device_generation is not None
                and device.registration.device_generation != item.device_generation
            )
            or (minimum_epoch is not None and item.agent_epoch < minimum_epoch)
            or (
                minimum_generation is not None
                and item.device_generation is not None
                and item.device_generation < minimum_generation
            )
        ):
            return OperationResult(ResultCode.STALE_TOKEN, "stale device token")
        if item.token_kind is IOTokenKind.WRITE:
            record = self._reservations.get(item.token_id)
            if record is None:
                # A successful commit consumes its reservation and is itself
                # proof that the write was terminal.
                return OperationResult(ResultCode.OK, "already committed or reclaimed")
            if (
                record.public.device_id != item.device_id
                or record.public.agent_epoch != item.agent_epoch
                or record.public.generation != item.extent_generation
                or (
                    item.device_generation is not None
                    and record.public.device_generation != item.device_generation
                )
            ):
                return OperationResult(ResultCode.STALE_TOKEN, "stale write token")
            record.write_terminal = True
            self._try_reclaim_reservation(record)
            return OperationResult(ResultCode.OK)
        if item.token_kind is IOTokenKind.READ:
            record = self._leases.get(item.token_id)
            if record is None:
                return OperationResult(ResultCode.OK, "already released and reclaimed")
            public = record.public
            if (
                public.device_id != item.device_id
                or public.agent_epoch != item.agent_epoch
                or public.extent_generation != item.extent_generation
                or (
                    item.device_generation is not None
                    and public.device_generation != item.device_generation
                )
            ):
                return OperationResult(ResultCode.STALE_TOKEN, "stale read token")
            record.io_terminal = True
            self._cleanup_lease_if_safe(record)
            return OperationResult(ResultCode.OK)
        return OperationResult(ResultCode.INVALID_ARGUMENT, "unknown token kind")

    def _maintenance(self, *, reap_only: bool = False) -> None:
        now = self.config.clock()
        self._refresh_device_health(now)
        self._prune_idempotency(now)
        for reservation in list(self._reservations.values()):
            if reservation.state == "reserved" and reservation.public.expires_at <= now:
                self._mark_reservation_reclaim_pending(reservation)
            self._try_reclaim_reservation(reservation)
        for lease in list(self._leases.values()):
            if lease.active and lease.public.expires_at <= now:
                lease.active = False
            self._cleanup_lease_if_safe(lease)
        self._advance_device_lifecycle(now)
        if not reap_only and self.config.auto_evict:
            for device_id in list(self._devices):
                if self._devices[device_id].state is not DeviceState.ONLINE:
                    continue
                report = self._capacity(self._devices[device_id])
                logical = report.reserved_bytes + report.ready_bytes
                if (
                    logical
                    > self._devices[device_id].high_watermark * report.capacity_bytes
                ):
                    self.run_eviction(device_id)

    def _begin_device_stop(self, device: _Device, grace_period_s: float) -> None:
        now = self.config.clock()
        device.state = DeviceState.DRAINING
        device.healthy = False
        device.drain_started_at = now
        device.drain_deadline = now + grace_period_s
        self._metrics["device_stops"] += 1

        device_id = device.registration.device_id
        for reservation in list(self._reservations.values()):
            if reservation.public.device_id == device_id:
                self._mark_reservation_reclaim_pending(reservation)
                self._try_reclaim_reservation(reservation)
        for obj in list(self._objects_by_generation.values()):
            if obj.mapping.locations[0].device_id == device_id:
                self._mark_object_reclaim_pending(obj)

    def _advance_device_lifecycle(self, now: float) -> None:
        for device in list(self._devices.values()):
            if device.state is not DeviceState.DRAINING:
                continue
            reservations, leases, _ = self._device_outstanding(device)
            if reservations == 0 and leases == 0:
                self._transition_device_offline(device, now, forced=False)
            elif device.drain_deadline is not None and now >= device.drain_deadline:
                self._transition_device_offline(device, now, forced=True)

    def _transition_device_offline(
        self, device: _Device, now: float, *, forced: bool
    ) -> None:
        device_id = device.registration.device_id

        for reservation_id, reservation in list(self._reservations.items()):
            if reservation.public.device_id != device_id:
                continue
            if self._reservation_by_key.get(reservation.key) == reservation_id:
                self._reservation_by_key.pop(reservation.key, None)
            self._reservations.pop(reservation_id, None)

        for lease_id, lease in list(self._leases.items()):
            if lease.public.device_id == device_id:
                self._leases.pop(lease_id, None)

        for generation, obj in list(self._objects_by_generation.items()):
            if obj.mapping.locations[0].device_id != device_id:
                continue
            if self._objects_by_key.get(obj.key) is obj:
                self._objects_by_key.pop(obj.key, None)
            self._objects_by_generation.pop(generation, None)

        device.free = [(0, device.registration.capacity_bytes)]
        device.state = DeviceState.OFFLINE
        device.healthy = False
        device.fenced = True
        device.drain_deadline = None
        device.offline_at = now
        self._minimum_agent_epoch[device_id] = max(
            self._minimum_agent_epoch.get(device_id, 1),
            device.registration.agent_epoch + 1,
        )
        self._minimum_device_generation[device_id] = max(
            self._minimum_device_generation.get(device_id, 1),
            device.registration.device_generation + 1,
        )
        if forced:
            self._metrics["device_forced_offlines"] += 1

    def _device_outstanding(self, device: _Device) -> tuple[int, int, int]:
        device_id = device.registration.device_id
        reservations = [
            record
            for record in self._reservations.values()
            if record.public.device_id == device_id
        ]
        leases = [
            record
            for record in self._leases.values()
            if record.public.device_id == device_id
        ]
        outstanding_io = sum(not record.write_terminal for record in reservations)
        outstanding_io += sum(not record.io_terminal for record in leases)
        return len(reservations), len(leases), outstanding_io

    def _mark_reservation_reclaim_pending(self, record: _ReservationRecord) -> None:
        if record.state == "reserved":
            record.state = "reclaim_pending"
            self._reservation_by_key.pop(record.key, None)

    def _try_reclaim_reservation(self, record: _ReservationRecord) -> bool:
        if record.state != "reclaim_pending" or not record.write_terminal:
            return False
        device = self._devices.get(record.public.device_id)
        if device is None:
            return False
        self._free(device, record.public.offset, record.public.allocated_length)
        self._reservations.pop(record.public.reservation_id, None)
        self._metrics["reclaimed_extents"] += 1
        return True

    def _try_reclaim_all_reservations(self) -> None:
        for record in list(self._reservations.values()):
            self._try_reclaim_reservation(record)

    def _mark_object_reclaim_pending(self, obj: _ObjectRecord) -> None:
        if obj.state == "ready":
            obj.state = "reclaim_pending"
            self._objects_by_key.pop(obj.key, None)
        self._try_reclaim_object(obj)

    def _try_reclaim_object(self, obj: _ObjectRecord) -> bool:
        if obj.state != "reclaim_pending" or not obj.write_terminal:
            return False
        for lease_id in list(obj.lease_ids):
            lease = self._leases.get(lease_id)
            if lease is None:
                obj.lease_ids.discard(lease_id)
                continue
            if lease.active or not lease.io_terminal:
                return False
            self._leases.pop(lease_id, None)
            obj.lease_ids.discard(lease_id)
        location = obj.mapping.locations[0]
        device = self._devices.get(location.device_id)
        if device is None:
            return False
        self._free(device, location.offset, location.allocated_length)
        self._objects_by_generation.pop(obj.mapping.object_generation, None)
        self._metrics["reclaimed_extents"] += 1
        return True

    def _cleanup_lease_if_safe(self, lease: _LeaseRecord) -> None:
        if lease.active or not lease.io_terminal:
            return
        obj = self._objects_by_generation.get(lease.object_generation)
        if obj is None:
            self._leases.pop(lease.public.lease_id, None)
            return
        # For a READY object retain no completed lease state; any subsequent
        # eviction cannot race an already-terminal read.
        obj.lease_ids.discard(lease.public.lease_id)
        self._leases.pop(lease.public.lease_id, None)
        if obj.state == "reclaim_pending":
            self._try_reclaim_object(obj)

    def _capacity(self, device: _Device) -> CapacityReport:
        device_id = device.registration.device_id
        reserved = sum(
            r.public.allocated_length
            for r in self._reservations.values()
            if r.public.device_id == device_id and r.state == "reserved"
        )
        pending_reservations = sum(
            r.public.allocated_length
            for r in self._reservations.values()
            if r.public.device_id == device_id and r.state == "reclaim_pending"
        )
        ready = 0
        pending_objects = 0
        for obj in self._objects_by_generation.values():
            location = obj.mapping.locations[0]
            if location.device_id != device_id:
                continue
            if obj.state == "ready":
                ready += location.allocated_length
            else:
                pending_objects += location.allocated_length
        return CapacityReport(
            device_id,
            device.registration.capacity_bytes,
            sum(length for _, length in device.free),
            reserved,
            ready,
            pending_reservations + pending_objects,
            max((length for _, length in device.free), default=0),
            device.low_watermark,
            device.high_watermark,
        )

    def _placement_order(self, preferred: Optional[int]) -> list[_Device]:
        self._refresh_device_health()
        devices = [
            d
            for d in self._devices.values()
            if d.healthy and d.state is DeviceState.ONLINE
        ]
        devices.sort(
            key=lambda d: (
                (
                    d.registration.device_id != preferred
                    if preferred is not None
                    else False
                ),
                -self._capacity(d).largest_free_extent,
                d.registration.device_id,
            )
        )
        return devices

    @staticmethod
    def _allocate(device: _Device, length: int) -> Optional[int]:
        alignment = device.registration.allocation_alignment
        for index, (start, available) in enumerate(device.free):
            aligned = _aligned_size(start, alignment)
            padding = aligned - start
            if padding + length > available:
                continue
            replacement = []
            if padding:
                replacement.append((start, padding))
            suffix_start = aligned + length
            suffix = start + available - suffix_start
            if suffix:
                replacement.append((suffix_start, suffix))
            device.free[index : index + 1] = replacement
            return aligned
        return None

    @staticmethod
    def _free(device: _Device, offset: int, length: int) -> None:
        intervals = sorted(device.free + [(offset, length)])
        merged: list[tuple[int, int]] = []
        for start, size in intervals:
            if not merged:
                merged.append((start, size))
                continue
            previous_start, previous_size = merged[-1]
            previous_end = previous_start + previous_size
            if start < previous_end:
                raise AssertionError("attempted to free an overlapping extent")
            if start == previous_end:
                merged[-1] = (previous_start, previous_size + size)
            else:
                merged.append((start, size))
        device.free = merged

    def _idempotent(self, operation, item, call, error_factory):
        identity_error = self._identity_error(item)
        if identity_error is not None:
            return error_factory(ResultCode.INVALID_ARGUMENT, identity_error)
        key = (operation, item.client_id, item.request_id)
        fingerprint = self._fingerprint(item)
        now = self.config.clock()
        cached = self._idempotency.get(key)
        if (
            cached is not None
            and cached.created_at > now - self.config.idempotency_ttl_s
        ):
            if cached.fingerprint != fingerprint:
                return error_factory(
                    ResultCode.CONFLICT,
                    "request identity was reused with different arguments",
                )
            self._metrics["idempotency_hits"] += 1
            return self._replay_cached(operation, item, cached.result, error_factory)
        if cached is not None:
            self._idempotency.pop(key, None)
            self._metrics["idempotency_evictions"] += 1
        self._prune_idempotency(now, reserve=1)
        result = call()
        self._idempotency[key] = _CachedResult(
            fingerprint, copy.deepcopy(result), self.config.clock()
        )
        return result

    def _prune_idempotency(self, now: float, *, reserve: int = 0) -> None:
        cutoff = now - self.config.idempotency_ttl_s
        removed = 0
        for key, cached in list(self._idempotency.items()):
            if cached.created_at <= cutoff:
                self._idempotency.pop(key, None)
                removed += 1

        target = max(0, self.config.max_idempotency_entries - reserve)
        overflow = len(self._idempotency) - target
        if overflow > 0:
            oldest = sorted(
                self._idempotency.items(), key=lambda entry: entry[1].created_at
            )
            for key, _ in oldest[:overflow]:
                self._idempotency.pop(key, None)
                removed += 1
        self._metrics["idempotency_evictions"] += removed

    def _replay_cached(self, operation, item, result, error_factory):
        """Replay ordinary results, but never replay a consumed I/O capability."""

        if operation == "reserve" and isinstance(result, ReserveResult):
            if result.reservation is not None:
                public = result.reservation
                record = self._reservations.get(public.reservation_id)
                if (
                    record is not None
                    and record.state == "reserved"
                    and record.public == public
                    and public.service_epoch == self._epoch
                ):
                    return copy.deepcopy(result)
                current = self._objects_by_key.get(item.key)
                if current is not None and current.state == "ready":
                    return ReserveResult(
                        ResultCode.EXISTING_READY,
                        "request already committed",
                        mapping=copy.deepcopy(current.mapping),
                    )
                return error_factory(
                    ResultCode.STALE_TOKEN,
                    "cached reservation capability is no longer active",
                )
            if result.mapping is not None:
                current = self._objects_by_key.get(item.key)
                if (
                    current is None
                    or current.state != "ready"
                    or current.mapping.object_generation
                    != result.mapping.object_generation
                ):
                    return error_factory(
                        ResultCode.STALE_TOKEN,
                        "cached object mapping is no longer current",
                    )

        if operation == "lookup" and isinstance(result, LookupResult):
            if result.lease is not None and result.mapping is not None:
                lease = self._leases.get(result.lease.lease_id)
                current = self._objects_by_key.get(item.key)
                if (
                    lease is None
                    or not lease.active
                    or lease.io_terminal
                    or lease.public != result.lease
                    or result.lease.service_epoch != self._epoch
                    or current is None
                    or current.state != "ready"
                    or current.mapping.object_generation
                    != result.mapping.object_generation
                ):
                    return error_factory(
                        ResultCode.STALE_TOKEN,
                        "cached read capability is no longer active",
                    )

        return copy.deepcopy(result)

    @staticmethod
    def _fingerprint(item: object) -> str:
        values = asdict(item)
        values.pop("deadline", None)
        return hashlib.sha256(repr(values).encode()).hexdigest()

    @staticmethod
    def _identity_error(item: object) -> Optional[str]:
        if not getattr(item, "client_id", ""):
            return "client_id must be non-empty"
        if not getattr(item, "request_id", ""):
            return "request_id must be non-empty"
        return None

    def _validate_identity_and_deadline(self, item) -> Optional[tuple[ResultCode, str]]:
        identity_error = self._identity_error(item)
        if identity_error:
            return ResultCode.INVALID_ARGUMENT, identity_error
        deadline = getattr(item, "deadline", None)
        if deadline is not None and self.config.clock() > deadline:
            return ResultCode.DEADLINE_EXCEEDED, "item deadline expired"
        return None

    def _preflight(
        self,
        items: Sequence[object],
        deadline: Optional[float],
        batch_request_id: str,
    ) -> None:
        if not isinstance(batch_request_id, str):
            raise BatchRejectedError("batch request ID must be a string")
        if not items:
            raise BatchRejectedError("batch must contain at least one item")
        if len(items) > self.config.max_batch_items:
            raise BatchRejectedError("batch item limit exceeded")
        encoded_size = len(batch_request_id.encode()) + sum(
            len(repr(item).encode()) for item in items
        )
        if encoded_size > self.config.max_batch_bytes:
            raise BatchRejectedError("batch byte limit exceeded")
        if deadline is not None and self.config.clock() > deadline:
            raise BatchRejectedError("batch deadline expired")

    def _validate_reservation_token(
        self, record: _ReservationRecord, service_epoch: int, generation: int
    ) -> Optional[ResultCode]:
        if service_epoch != self._epoch or service_epoch != record.public.service_epoch:
            return ResultCode.STALE_TOKEN
        if generation != record.public.generation:
            return ResultCode.STALE_TOKEN
        return None

    @staticmethod
    def _key_digest(key: bytes) -> str:
        return hashlib.sha256(key).hexdigest()

    @staticmethod
    def _validate_registration(registration: DeviceRegistration) -> None:
        if registration.device_id < 0:
            raise ValueError("device ID must be non-negative")
        if not registration.agent_endpoint:
            raise ValueError("agent endpoint must be non-empty")
        if registration.capacity_bytes <= 0:
            raise ValueError("capacity must be positive")
        alignment = registration.allocation_alignment
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("allocation alignment must be a power of two")
        if registration.capacity_bytes % alignment:
            raise ValueError("capacity must be alignment-sized")
        if registration.agent_epoch <= 0:
            raise ValueError("agent epoch must be positive")
        if registration.device_generation <= 0:
            raise ValueError("device generation must be positive")

    def _refresh_device_health(self, now: Optional[float] = None) -> None:
        now = self.config.clock() if now is None else now
        for device in self._devices.values():
            if (
                device.state is DeviceState.ONLINE
                and now - device.last_heartbeat > self.config.device_timeout_s
            ):
                device.healthy = False

    def _device_info(self, device: _Device) -> DeviceInfo:
        registration = device.registration
        reservations, leases, outstanding_io = self._device_outstanding(device)
        return DeviceInfo(
            registration.device_id,
            registration.agent_endpoint,
            registration.capacity_bytes,
            registration.allocation_alignment,
            registration.agent_epoch,
            registration.numa_node,
            registration.failure_domain,
            device.healthy,
            device.last_heartbeat,
            registration.device_generation,
            device.state,
            device.drain_started_at,
            device.offline_at,
            reservations,
            leases,
            outstanding_io,
        )
