# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest
from nixl_shard.service import (
    AbortItem,
    BatchRejectedError,
    CommitItem,
    DeviceRegistration,
    DeviceState,
    IoTerminalReport,
    IOTokenKind,
    LookupItem,
    ReleaseReadItem,
    ReserveItem,
    ResultCode,
    ServiceConfig,
    ShardNamingService,
    TouchItem,
)

PAGE = 4096


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _service(pages=8, **config_overrides):
    clock = config_overrides.pop("clock", FakeClock())
    config = ServiceConfig(clock=clock, auto_evict=False, **config_overrides)
    service = ShardNamingService(config, epoch=1)
    result = service.register_device(
        DeviceRegistration(7, "127.0.0.1:7007", pages * PAGE, PAGE, 11)
    )
    assert result.code is ResultCode.OK
    return service, clock


def _reserve(service, key, request_id, length=1, ttl=None):
    [result] = service.batch_reserve(
        [
            ReserveItem(
                key,
                length,
                "test-client",
                request_id,
                reservation_ttl_s=ttl,
            )
        ]
    )
    assert result.code is ResultCode.OK
    return result.reservation


def _commit(service, reservation, request_id):
    [result] = service.batch_commit(
        [
            CommitItem(
                reservation.reservation_id,
                reservation.generation,
                reservation.service_epoch,
                "test-client",
                request_id,
            )
        ]
    )
    assert result.code is ResultCode.OK
    return result.mapping


def _write_terminal(service, reservation, request_id):
    return service.report_io_terminal(
        IoTerminalReport(
            reservation.reservation_id,
            IOTokenKind.WRITE,
            reservation.service_epoch,
            reservation.device_id,
            reservation.agent_epoch,
            reservation.generation,
            "test-client",
            request_id,
        )
    )


def _read_terminal(service, lease, request_id):
    return service.report_io_terminal(
        IoTerminalReport(
            lease.lease_id,
            IOTokenKind.READ,
            lease.service_epoch,
            lease.device_id,
            lease.agent_epoch,
            lease.extent_generation,
            "test-client",
            request_id,
        )
    )


def test_reserved_is_invisible_commit_publishes_and_immutable_put_converges():
    service, _ = _service()
    reservation = _reserve(service, b"page-a", "reserve-a")

    [miss] = service.batch_lookup([LookupItem(b"page-a", "reader", "lookup-a")])
    assert miss.code is ResultCode.NOT_FOUND

    mapping = _commit(service, reservation, "commit-a")
    [hit] = service.batch_lookup([LookupItem(b"page-a", "reader", "lookup-b")])
    assert hit.code is ResultCode.OK
    assert hit.mapping == mapping

    [duplicate] = service.batch_reserve(
        [ReserveItem(b"page-a", 1, "writer-2", "reserve-duplicate")]
    )
    assert duplicate.code is ResultCode.EXISTING_READY
    assert duplicate.mapping == mapping
    service.validate_invariants()


def test_batch_exists_is_read_only_and_does_not_reserve_future_get():
    service, _ = _service(pages=2)
    reservation = _reserve(service, b"ready", "reserve-ready")
    assert service.batch_exists([b"ready", b"missing"]) == [False, False]
    _commit(service, reservation, "commit-ready")

    before = service.get_metrics()
    assert service.batch_exists([b"ready", b"missing", b"ready"]) == [
        True,
        False,
        True,
    ]
    after = service.get_metrics()
    assert after["leases_active"] == before["leases_active"] == 0
    assert after["lookups"] == before["lookups"]
    assert after["lookup_misses"] == before["lookup_misses"]

    # Exists is a snapshot, not a lease. A stopped device can invalidate the
    # object before the subsequent GET lookup.
    service.stop_device(7, 11)
    assert service.batch_exists([b"ready"]) == [False]
    [miss] = service.batch_lookup([LookupItem(b"ready", "reader", "after-stop")])
    assert miss.code is ResultCode.NOT_FOUND


def test_batch_exists_preserves_batch_limits_and_key_validation():
    service, _ = _service(max_batch_items=2)
    with pytest.raises(BatchRejectedError, match="item limit"):
        service.batch_exists([b"a", b"b", b"c"])
    with pytest.raises(BatchRejectedError, match="non-empty bytes"):
        service.batch_exists([b""])


def test_batches_are_ordered_partial_and_idempotent_with_preflight_limits():
    service, _ = _service(pages=4, max_batch_items=3)
    items = [
        ReserveItem(b"a", 1, "client", "one"),
        ReserveItem(b"bad", 0, "client", "two"),
        ReserveItem(b"c", PAGE + 1, "client", "three"),
    ]
    results = service.batch_reserve(items, batch_request_id="batch-1")
    assert [result.code for result in results] == [
        ResultCode.OK,
        ResultCode.INVALID_ARGUMENT,
        ResultCode.OK,
    ]
    assert results[0].reservation.offset == 0
    assert results[2].reservation.offset == PAGE

    retry = service.batch_reserve([items[0]])[0]
    assert retry == results[0]
    [conflict] = service.batch_reserve([ReserveItem(b"different", 1, "client", "one")])
    assert conflict.code is ResultCode.CONFLICT

    before = service.get_capacity(7)
    with pytest.raises(BatchRejectedError, match="item limit"):
        service.batch_reserve(items + [ReserveItem(b"d", 1, "client", "four")])
    assert service.get_capacity(7) == before
    assert service.get_metrics()["idempotency_hits"] == 1
    service.validate_invariants()

    byte_limited, _ = _service(pages=1, max_batch_bytes=8)
    before = byte_limited.get_capacity(7)
    with pytest.raises(BatchRejectedError, match="byte limit"):
        byte_limited.batch_reserve(
            [ReserveItem(b"too-large", 1, "client", "byte-limited")]
        )
    assert byte_limited.get_capacity(7) == before


def test_abort_and_expiry_wait_for_explicit_write_terminal_before_reuse():
    service, clock = _service(pages=1)
    first = _reserve(service, b"first", "reserve-first")
    [aborted] = service.batch_abort(
        [
            AbortItem(
                first.reservation_id,
                first.generation,
                first.service_epoch,
                "test-client",
                "abort-first",
            )
        ]
    )
    assert aborted.code is ResultCode.OK
    capacity = service.get_capacity(7)
    assert capacity.reclaim_pending_bytes == PAGE
    assert capacity.free_bytes == 0
    [no_space] = service.batch_reserve(
        [ReserveItem(b"blocked", 1, "test-client", "reserve-blocked")]
    )
    assert no_space.code is ResultCode.NO_SPACE

    assert _write_terminal(service, first, "terminal-first").code is ResultCode.OK
    assert service.get_capacity(7).free_bytes == PAGE

    expiring = _reserve(service, b"expiring", "reserve-expiring", ttl=1)
    clock.advance(2)
    service.reap_expired()
    capacity = service.get_capacity(7)
    assert capacity.reclaim_pending_bytes == PAGE
    assert capacity.free_bytes == 0
    [late_commit] = service.batch_commit(
        [
            CommitItem(
                expiring.reservation_id,
                expiring.generation,
                expiring.service_epoch,
                "test-client",
                "late-commit",
            )
        ]
    )
    assert late_commit.code is ResultCode.STALE_TOKEN
    assert _write_terminal(service, expiring, "terminal-expired").code is ResultCode.OK
    assert service.get_capacity(7).free_bytes == PAGE
    service.validate_invariants()


def test_read_release_and_terminal_are_both_required_for_evicted_extent():
    service, _ = _service(pages=1)
    reservation = _reserve(service, b"object", "reserve")
    mapping = _commit(service, reservation, "commit")
    [lookup] = service.batch_lookup([LookupItem(b"object", "reader", "lookup")])
    lease = lookup.lease

    eviction = service.run_eviction(7, force=True)
    assert eviction.selected_objects == 1
    assert service.get_capacity(7).reclaim_pending_bytes == PAGE
    [miss] = service.batch_lookup([LookupItem(b"object", "reader", "post-evict")])
    assert miss.code is ResultCode.NOT_FOUND

    [released] = service.batch_release_read(
        [
            ReleaseReadItem(
                lease.lease_id,
                mapping.object_generation,
                lease.service_epoch,
                "reader",
                "release",
            )
        ]
    )
    assert released.code is ResultCode.OK
    assert service.get_capacity(7).free_bytes == 0

    assert _read_terminal(service, lease, "read-terminal").code is ResultCode.OK
    capacity = service.get_capacity(7)
    assert capacity.free_bytes == PAGE
    assert capacity.reclaim_pending_bytes == 0
    service.validate_invariants()


def test_expired_read_lease_still_waits_for_terminal_io():
    service, clock = _service(pages=1)
    _commit(service, _reserve(service, b"object", "r"), "c")
    [lookup] = service.batch_lookup(
        [LookupItem(b"object", "reader", "l", lease_ttl_s=1)]
    )
    service.run_eviction(7, force=True)
    clock.advance(2)
    service.reap_expired()
    assert service.get_capacity(7).reclaim_pending_bytes == PAGE
    assert _read_terminal(service, lookup.lease, "terminal").code is ResultCode.OK
    assert service.get_capacity(7).free_bytes == PAGE


def test_terminal_read_still_waits_for_active_lease_release():
    service, _ = _service(pages=1)
    mapping = _commit(service, _reserve(service, b"object", "r"), "c")
    [lookup] = service.batch_lookup([LookupItem(b"object", "reader", "l")])
    assert _read_terminal(service, lookup.lease, "terminal").code is ResultCode.OK
    service.run_eviction(7, force=True)
    assert service.get_capacity(7).free_bytes == 0
    [released] = service.batch_release_read(
        [
            ReleaseReadItem(
                lookup.lease.lease_id,
                mapping.object_generation,
                lookup.lease.service_epoch,
                "reader",
                "release",
            )
        ]
    )
    assert released.code is ResultCode.OK
    assert service.get_capacity(7).free_bytes == PAGE


def test_eviction_reaches_low_watermark_and_touch_changes_lru():
    service, clock = _service(pages=5)
    mappings = {}
    for index in range(5):
        key = f"key-{index}".encode()
        mappings[key] = _commit(
            service,
            _reserve(service, key, f"reserve-{index}"),
            f"commit-{index}",
        )
        clock.advance(1)
    assert service.set_watermarks(7, 0.4, 0.8).code is ResultCode.OK
    clock.advance(1)
    [touched] = service.batch_touch([TouchItem(b"key-0", "reader", "touch-key-zero")])
    assert touched.code is ResultCode.OK

    result = service.run_eviction(7)
    assert result.selected_objects == 3
    capacity = service.get_capacity(7)
    assert capacity.ready_bytes == 2 * PAGE
    assert capacity.free_bytes == 3 * PAGE
    assert service.get_object_debug(b"key-0")["state"] == "ready"
    service.validate_invariants()


def test_randomized_allocator_never_overlaps_and_coalesces():
    service, _ = _service(pages=32)
    reservations = []
    random_source = random.Random(98123)
    for index in range(200):
        if reservations and random_source.random() < 0.45:
            reservation = reservations.pop(random_source.randrange(len(reservations)))
            _write_terminal(service, reservation, f"terminal-{index}")
            [result] = service.batch_abort(
                [
                    AbortItem(
                        reservation.reservation_id,
                        reservation.generation,
                        reservation.service_epoch,
                        "test-client",
                        f"abort-{index}",
                    )
                ]
            )
            assert result.code is ResultCode.OK
        else:
            [result] = service.batch_reserve(
                [
                    ReserveItem(
                        f"key-{index}".encode(),
                        random_source.randint(1, 3 * PAGE),
                        "random",
                        f"reserve-{index}",
                    )
                ]
            )
            if result.code is ResultCode.OK:
                reservations.append(result.reservation)
            else:
                assert result.code is ResultCode.NO_SPACE
        service.validate_invariants()
    for index, reservation in enumerate(reservations):
        _write_terminal(service, reservation, f"final-terminal-{index}")
        service.batch_abort(
            [
                AbortItem(
                    reservation.reservation_id,
                    reservation.generation,
                    reservation.service_epoch,
                    "test-client",
                    f"final-abort-{index}",
                )
            ]
        )
    capacity = service.get_capacity(7)
    assert capacity.free_bytes == capacity.capacity_bytes
    assert capacity.largest_free_extent == capacity.capacity_bytes


def test_device_collision_fencing_and_higher_epoch_registration():
    service, _ = _service(pages=2)
    conflict = service.register_device(
        DeviceRegistration(7, "127.0.0.1:9999", 2 * PAGE, PAGE, 12)
    )
    assert conflict.code is ResultCode.CONFLICT
    _commit(service, _reserve(service, b"old", "reserve-old"), "commit-old")

    assert service.fence_device_epoch(7, 11, quiesced=False).code is ResultCode.OK
    assert service.heartbeat(7, 11).code is ResultCode.STALE_TOKEN
    assert service.fence_device_epoch(7, 11, quiesced=True).code is ResultCode.OK
    stale_registration = service.register_device(
        DeviceRegistration(7, "127.0.0.1:7007", 2 * PAGE, PAGE, 11)
    )
    assert stale_registration.code is ResultCode.STALE_TOKEN
    replacement = service.register_device(
        DeviceRegistration(
            7,
            "127.0.0.1:9999",
            2 * PAGE,
            PAGE,
            12,
            device_generation=2,
        )
    )
    assert replacement.code is ResultCode.OK
    assert service.get_capacity(7).free_bytes == 2 * PAGE
    service.validate_invariants()


def test_service_restart_discards_mappings_and_rejects_old_epoch_tokens():
    service, _ = _service(pages=2)
    reservation = _reserve(service, b"old", "reserve-old")
    assert service.restart() == 2
    assert service.health()["service_epoch"] == 2
    service.register_device(
        DeviceRegistration(
            7,
            "127.0.0.1:7007",
            2 * PAGE,
            PAGE,
            12,
            device_generation=2,
        )
    )
    [stale] = service.batch_commit(
        [
            CommitItem(
                reservation.reservation_id,
                reservation.generation,
                reservation.service_epoch,
                "test-client",
                "stale-commit",
            )
        ]
    )
    assert stale.code is ResultCode.STALE_TOKEN
    [miss] = service.batch_lookup([LookupItem(b"old", "reader", "miss")])
    assert miss.code is ResultCode.NOT_FOUND
    assert service.get_capacity(7).free_bytes == 2 * PAGE

    fresh = ShardNamingService(ServiceConfig(auto_evict=False))
    assert fresh.epoch != service.epoch
    fresh.register_device(DeviceRegistration(7, "127.0.0.1:7007", 2 * PAGE, PAGE, 13))
    [stale_after_process_restart] = fresh.batch_commit(
        [
            CommitItem(
                reservation.reservation_id,
                reservation.generation,
                reservation.service_epoch,
                "test-client",
                "fresh-process-stale-commit",
            )
        ]
    )
    assert stale_after_process_restart.code is ResultCode.STALE_TOKEN


def test_stop_device_blocks_new_io_drains_and_requires_new_generation():
    service, _ = _service(pages=2)
    mapping = _commit(service, _reserve(service, b"ready", "reserve-ready"), "commit")
    [lookup] = service.batch_lookup(
        [LookupItem(b"ready", "reader", "lookup-before-stop")]
    )
    reservation = _reserve(service, b"writing", "reserve-writing")

    stopped = service.stop_device(7, 11, device_generation=1, grace_period_s=10)
    assert stopped.code is ResultCode.OK
    assert stopped.device.state is DeviceState.DRAINING
    assert stopped.device.healthy is False
    assert stopped.device.outstanding_reservations == 1
    assert stopped.device.outstanding_read_leases == 1
    assert stopped.device.outstanding_io == 2

    [no_placement] = service.batch_reserve(
        [ReserveItem(b"new", 1, "client", "reserve-after-stop")]
    )
    assert no_placement.code is ResultCode.UNAVAILABLE
    [hidden] = service.batch_lookup(
        [LookupItem(b"ready", "reader", "lookup-after-stop")]
    )
    assert hidden.code is ResultCode.NOT_FOUND
    assert service.heartbeat(7, 11).code is ResultCode.UNAVAILABLE
    [late_commit] = service.batch_commit(
        [
            CommitItem(
                reservation.reservation_id,
                reservation.generation,
                reservation.service_epoch,
                "test-client",
                "commit-after-stop",
            )
        ]
    )
    assert late_commit.code is ResultCode.STALE_TOKEN

    [released] = service.batch_release_read(
        [
            ReleaseReadItem(
                lookup.lease.lease_id,
                mapping.object_generation,
                lookup.lease.service_epoch,
                "reader",
                "release-after-stop",
            )
        ]
    )
    assert released.code is ResultCode.OK
    assert (
        _read_terminal(service, lookup.lease, "read-terminal-after-stop").code
        is ResultCode.OK
    )
    assert service.get_device_status(7).device.state is DeviceState.DRAINING
    assert (
        _write_terminal(service, reservation, "write-terminal-after-stop").code
        is ResultCode.OK
    )

    offline = service.get_device_status(7).device
    assert offline.state is DeviceState.OFFLINE
    assert offline.outstanding_reservations == 0
    assert offline.outstanding_read_leases == 0
    assert offline.outstanding_io == 0
    assert service.get_capacity(7).free_bytes == 2 * PAGE

    assert (
        service.register_device(
            DeviceRegistration(
                7,
                "127.0.0.1:7008",
                2 * PAGE,
                PAGE,
                12,
                device_generation=1,
            )
        ).code
        is ResultCode.STALE_TOKEN
    )
    assert (
        service.register_device(
            DeviceRegistration(
                7,
                "127.0.0.1:7008",
                2 * PAGE,
                PAGE,
                11,
                device_generation=2,
            )
        ).code
        is ResultCode.STALE_TOKEN
    )
    replacement = service.register_device(
        DeviceRegistration(
            7,
            "127.0.0.1:7008",
            2 * PAGE,
            PAGE,
            12,
            device_generation=2,
        )
    )
    assert replacement.code is ResultCode.OK
    assert replacement.device.state is DeviceState.ONLINE
    assert replacement.device.device_generation == 2
    service.validate_invariants()


def test_stop_device_grace_forces_offline_and_fences_old_tokens():
    clock = FakeClock()
    service, _ = _service(pages=2, clock=clock, device_stop_grace_s=2)
    mapping = _commit(service, _reserve(service, b"ready", "reserve-ready"), "commit")
    [lookup] = service.batch_lookup([LookupItem(b"ready", "reader", "lookup")])
    reservation = _reserve(service, b"writing", "reserve-writing")

    stopped = service.drain_device(7, 11, device_generation=1)
    assert stopped.device.state is DeviceState.DRAINING
    clock.advance(3)
    service.reap_expired()

    offline = service.get_device_status(7).device
    assert offline.state is DeviceState.OFFLINE
    assert offline.offline_at == clock.now
    assert service.get_metrics()["device_forced_offlines"] == 1
    assert service.get_object_debug(mapping.key_digest)["state"] == "not_found"

    replacement = service.register_device(
        DeviceRegistration(
            7,
            "127.0.0.1:7008",
            2 * PAGE,
            PAGE,
            12,
            device_generation=2,
        )
    )
    assert replacement.code is ResultCode.OK
    assert (
        _write_terminal(service, reservation, "old-write-terminal").code
        is ResultCode.STALE_TOKEN
    )
    assert (
        _read_terminal(service, lookup.lease, "old-read-terminal").code
        is ResultCode.STALE_TOKEN
    )
    assert service.get_capacity(7).free_bytes == 2 * PAGE
    service.validate_invariants()


def test_timed_out_device_can_heartbeat_but_fenced_epoch_cannot():
    service, clock = _service(pages=2, device_timeout_s=1)
    clock.advance(2)
    assert service.list_devices()[0].healthy is False
    [unavailable] = service.batch_reserve(
        [ReserveItem(b"unavailable", 1, "client", "before-heartbeat")]
    )
    assert unavailable.code is ResultCode.UNAVAILABLE
    assert service.heartbeat(7, 11).code is ResultCode.OK
    assert service.list_devices()[0].healthy is True
    assert (
        service.batch_reserve(
            [ReserveItem(b"available", 1, "client", "after-heartbeat")]
        )[0].code
        is ResultCode.OK
    )
    assert service.fence_device_epoch(7, 11, quiesced=False).code is ResultCode.OK
    assert service.heartbeat(7, 11).code is ResultCode.STALE_TOKEN


def test_automatic_high_watermark_eviction_reaches_low_watermark():
    clock = FakeClock()
    service = ShardNamingService(
        ServiceConfig(
            clock=clock, auto_evict=True, low_watermark=0.4, high_watermark=0.8
        ),
        epoch=1,
    )
    service.register_device(DeviceRegistration(7, "127.0.0.1:7007", 5 * PAGE, PAGE, 11))
    for index in range(5):
        key = f"auto-{index}".encode()
        _commit(
            service,
            _reserve(service, key, f"auto-reserve-{index}"),
            f"auto-commit-{index}",
        )
        clock.advance(1)
    capacity = service.get_capacity(7)
    assert capacity.ready_bytes == 2 * PAGE
    assert capacity.free_bytes == 3 * PAGE
    service.validate_invariants()


def test_idempotency_retention_is_bounded_and_expires():
    clock = FakeClock()
    service, _ = _service(
        clock=clock,
        max_idempotency_entries=2,
        idempotency_ttl_s=5,
    )
    for index in range(3):
        [result] = service.batch_touch(
            [TouchItem(b"missing", "client", f"touch-{index}")]
        )
        assert result.code is ResultCode.NOT_FOUND

    metrics = service.get_metrics()
    assert metrics["idempotency_entries"] == 2
    assert metrics["idempotency_evictions"] == 1

    clock.advance(6)
    service.reap_expired()
    metrics = service.get_metrics()
    assert metrics["idempotency_entries"] == 0
    assert metrics["idempotency_evictions"] == 3


def test_consumed_reservation_idempotency_never_replays_reused_extent():
    service, _ = _service(pages=1)
    request = ReserveItem(b"old", 1, "writer", "same-reserve")
    [reserved] = service.batch_reserve([request])
    assert reserved.code is ResultCode.OK
    _commit(service, reserved.reservation, "commit-old")

    [committed_retry] = service.batch_reserve([request])
    assert committed_retry.code is ResultCode.EXISTING_READY
    assert committed_retry.mapping.object_generation == 1

    assert service.run_eviction(7, force=True).reclaimed_bytes == PAGE
    replacement = _reserve(service, b"new", "reserve-new")
    assert replacement.offset == reserved.reservation.offset

    [stale_retry] = service.batch_reserve([request])
    assert stale_retry.code is ResultCode.STALE_TOKEN
    assert stale_retry.reservation is None
    service.validate_invariants()


def test_consumed_lookup_idempotency_never_aliases_reused_extent():
    service, _ = _service(pages=1)
    _commit(service, _reserve(service, b"old", "reserve-old"), "commit-old")
    request = LookupItem(b"old", "reader", "same-lookup")
    [lookup] = service.batch_lookup([request])
    assert lookup.code is ResultCode.OK
    assert _read_terminal(service, lookup.lease, "terminal-old").code is ResultCode.OK
    [released] = service.batch_release_read(
        [
            ReleaseReadItem(
                lookup.lease.lease_id,
                lookup.mapping.object_generation,
                lookup.lease.service_epoch,
                "reader",
                "release-old",
            )
        ]
    )
    assert released.code is ResultCode.OK
    assert service.run_eviction(7, force=True).reclaimed_bytes == PAGE
    replacement = _commit(
        service, _reserve(service, b"new", "reserve-new"), "commit-new"
    )

    [stale_retry] = service.batch_lookup([request])
    assert stale_retry.code is ResultCode.STALE_TOKEN
    assert stale_retry.mapping is None
    assert replacement.object_generation != lookup.mapping.object_generation
    service.validate_invariants()
