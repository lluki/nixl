# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import time

import pytest
from nixl_shard.agent import AgentConfig, ShardAgent, buffer_region_from_writable
from nixl_shard.bootstrap import create_client
from nixl_shard.client import (
    ClientConfig,
    GetItem,
    KVResult,
    KVStatus,
    SetItem,
    ShardClient,
)
from nixl_shard.service import (
    DeviceRegistration,
    ResultCode,
    ServiceConfig,
    ShardNamingService,
)


@pytest.fixture
def client_stack(tmp_path):
    device_path = tmp_path / "device.bin"
    service = ShardNamingService(ServiceConfig(auto_evict=False))
    agent = ShardAgent.open(
        AgentConfig(
            device_path,
            64 * 1024,
            logical_device_id=7,
            create=True,
            direct_io=False,
        )
    )
    registered = service.register_device(
        DeviceRegistration(
            device_id=7,
            agent_endpoint="inproc://device-7",
            capacity_bytes=agent.size,
            allocation_alignment=4096,
            agent_epoch=1,
            numa_node=2,
        )
    )
    assert registered.code is ResultCode.OK
    client = ShardClient(ClientConfig(service=service, agent=agent))
    try:
        yield client, service
    finally:
        client.close()
        agent.close(timeout=5)


def test_batched_set_get_exists_and_duplicate(client_stack):
    client, service = client_stack
    sources = [bytearray(b"alpha" * 73), bytearray(b"beta" * 91)]
    keys = [b"model/rank/page-1", b"model/rank/page-2"]

    stored = client.batch_set(keys, sources)
    assert [result.status for result in stored] == [KVStatus.OK, KVStatus.OK]
    assert service.get_metrics()["commits"] == 2
    assert client.get_metrics()["metadata_calls"] == 3

    duplicate = client.batch_set([keys[0]], [sources[0]])
    assert duplicate[0].status is KVStatus.ALREADY_PRESENT

    destinations = [bytearray(len(value)) for value in sources]
    loaded = client.batch_get(keys, destinations)
    assert [result.status for result in loaded] == [KVStatus.OK, KVStatus.OK]
    assert destinations == sources
    before = client.get_metrics()["metadata_calls"]
    assert client.batch_exists([keys[0], b"missing"]) == [True, False]
    assert client.get_metrics()["metadata_calls"] - before == 1
    page_keys = [keys[0]] + [f"missing/{index}".encode() for index in range(46)]
    before = client.get_metrics()["metadata_calls"]
    assert client.batch_exists(page_keys) == [True] + [False] * 46
    assert client.get_metrics()["metadata_calls"] - before == 1
    service.validate_invariants()
    assert service.get_metrics()["leases_active"] == 0


def test_get_trace_records_lookup_and_request_id(client_stack, monkeypatch, tmp_path):
    trace_path = tmp_path / "client-trace.jsonl"
    monkeypatch.setenv("NIXLSHARD_TRACE_PATH", str(trace_path))
    client, _ = client_stack
    source = bytearray(b"trace-value")
    assert client.batch_set([b"trace-key"], [source])[0].ok
    destination = bytearray(len(source))
    result = client.batch_get(
        [GetItem(b"trace-key", destination, request_id="trace-get")]
    )
    assert result[0].status is KVStatus.OK
    assert destination == source
    [event] = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert event["event"] == "client_get"
    assert event["request_ids"] == ["trace-get"]
    assert event["metadata_lookup_ns"] > 0
    assert event["client_e2e_ns"] >= event["metadata_lookup_ns"]
    assert event["register_memory_ns"] > 0
    assert event["wait_ns"] > 0
    assert event["terminal_report_ns"] > 0
    assert client.batch_exists([b"trace-key"]) == [True]
    get_event, exists_event = [
        json.loads(line) for line in trace_path.read_text().splitlines()
    ]
    assert exists_event["event"] == "client_exists"
    assert exists_event["key_digests"] == get_event["key_digests"]
    assert exists_event["metadata_lookup_ns"] > 0
    assert exists_event["terminal_report_ns"] == 0
    assert exists_event["release_lease_ns"] == 0


def test_ordered_mixed_results_and_numa_hint(client_stack):
    client, _ = client_stack
    assert client.get_device_id_for_numa_node(2) == 7
    values = [bytearray(b"one"), bytearray(b"two")]
    result = client.batch_set(
        [
            SetItem(b"valid", values[0], preferred_device_id=7),
            SetItem(b"", values[1]),
        ]
    )
    assert [item.status for item in result] == [
        KVStatus.OK,
        KVStatus.INVALID_ARGUMENT,
    ]

    targets = [bytearray(3), bytearray(3)]
    result = client.batch_get(
        [GetItem(b"valid", targets[0]), GetItem(b"absent", targets[1])]
    )
    assert [item.status for item in result] == [KVStatus.OK, KVStatus.MISS]
    assert targets[0] == values[0]


def test_async_batch_retains_buffers_until_wait(client_stack):
    client, _ = client_stack
    source = bytearray(b"async-value")
    set_handles = client.submit_batch_set([b"async"], [source])
    assert client.wait(set_handles, timeout=5)[0].status is KVStatus.OK

    destination = bytearray(len(source))
    get_handles = client.submit_batch_get([b"async"], [destination])
    completed = client.wait(get_handles, timeout=5)
    assert completed[0].status is KVStatus.OK
    assert destination == source


def test_lazy_release_is_bounded_and_close_drains_even_without_wait(
    client_stack, monkeypatch
):
    base, service = client_stack
    source = bytearray(b"lazy-value")
    assert base.batch_set([b"lazy-key"], [source])[0].ok
    client = ShardClient(
        ClientConfig(
            service=service,
            agent=base._agent,
            lazy_release=True,
            max_pending_lazy_releases=1,
        )
    )
    entered = threading.Event()
    release = threading.Event()
    original = service.batch_report_io_terminal
    calls = 0

    def block_first_report(items, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(5)
        return original(items, **kwargs)

    monkeypatch.setattr(service, "batch_report_io_terminal", block_first_report)
    try:
        for iteration in range(3):
            destination = bytearray(len(source))
            assert client.batch_get([b"lazy-key"], [destination])[0].ok
            assert destination == source
            if iteration == 0:
                assert entered.wait(5)
        metrics = client.get_metrics()
        assert metrics["lazy_release_enqueued"] == 2
        assert metrics["lazy_release_fallbacks"] == 1
        assert metrics["lazy_release_pending"] == 2
        # The first two leases remain pinned while the report is delayed.
        assert service.get_metrics()["leases_active"] == 2
        closer = threading.Thread(target=lambda: client.close(wait=False))
        closer.start()
        closer.join(timeout=0.05)
        assert closer.is_alive()
        release.set()
        closer.join(timeout=5)
        assert not closer.is_alive()
        assert client.get_metrics()["lazy_release_pending"] == 0
        assert client.get_metrics()["lazy_release_completed"] == 2
        assert service.get_metrics()["leases_active"] == 0
        service.validate_invariants()
    finally:
        release.set()
        client.close()


def test_lazy_release_reports_background_metadata_failure(client_stack, monkeypatch):
    base, service = client_stack
    source = bytearray(b"lazy-error")
    assert base.batch_set([b"lazy-error-key"], [source])[0].ok
    client = ShardClient(
        ClientConfig(service=service, agent=base._agent, lazy_release=True)
    )

    def fail_release(items, **kwargs):
        raise TimeoutError("release unavailable")

    monkeypatch.setattr(service, "batch_release_read", fail_release)
    try:
        destination = bytearray(len(source))
        assert client.batch_get([b"lazy-error-key"], [destination])[0].ok
        assert destination == source
        client.close()
        assert client.get_metrics()["lazy_release_failed"] == 1
        assert "release unavailable" in client.get_lazy_release_errors()[0]
    finally:
        client.close()


def test_destination_too_small_releases_lease(client_stack):
    client, service = client_stack
    assert client.batch_set([b"large"], [bytearray(b"0123456789")])[0].ok
    result = client.batch_get([b"large"], [bytearray(4)])
    assert result[0].status is KVStatus.RESOURCE_EXHAUSTED
    assert service.get_metrics()["leases_active"] == 0
    service.validate_invariants()


def test_one_key_maps_scatter_gather_page_to_one_contiguous_extent(client_stack):
    client, service = client_stack
    source_parts = [bytearray(b"key-component"), bytearray(b"value-component")]
    source = [buffer_region_from_writable(part) for part in source_parts]

    stored = client.batch_set([b"logical-page"], [source])
    assert stored[0].status is KVStatus.OK
    assert service.get_metrics()["objects_ready"] == 1

    destination_parts = [bytearray(len(part)) for part in source_parts]
    destination = [buffer_region_from_writable(part) for part in destination_parts]
    loaded = client.batch_get([b"logical-page"], [destination])
    assert loaded[0].status is KVStatus.OK
    assert destination_parts == source_parts
    service.validate_invariants()


def test_owned_device_heartbeat_keeps_registration_healthy(tmp_path):
    service = ShardNamingService(ServiceConfig(auto_evict=False, device_timeout_s=0.05))
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "heartbeat.bin",
            4096,
            logical_device_id=5,
            create=True,
            direct_io=False,
        )
    )
    registered = service.register_device(
        DeviceRegistration(5, "inproc://heartbeat", 4096, 4096, 9)
    )
    assert registered.code is ResultCode.OK
    client = ShardClient(
        ClientConfig(
            service=service,
            agent=agent,
            heartbeat_device_id=5,
            heartbeat_agent_epoch=9,
            heartbeat_interval_s=0.01,
        )
    )
    try:
        deadline = time.monotonic() + 1
        while client.get_metrics()["heartbeat_successes"] == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        time.sleep(0.08)
        assert service.list_devices()[0].healthy is True
    finally:
        client.close()
        assert service.get_device_status(5).device.state.value == "offline"
        agent.close(timeout=5)


def test_client_uses_logical_device_generation_not_agent_epoch(tmp_path):
    service = ShardNamingService(ServiceConfig(auto_evict=False))
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "generation.bin",
            4096,
            logical_device_id=12,
            logical_device_generation=7,
            create=True,
            direct_io=False,
        )
    )
    registered = service.register_device(
        DeviceRegistration(
            12,
            "inproc://generation",
            4096,
            4096,
            agent_epoch=42,
            device_generation=7,
        )
    )
    assert registered.code is ResultCode.OK
    client = ShardClient(ClientConfig(service=service, agent=agent))
    try:
        source = bytearray(b"generation-fenced")
        assert client.batch_set([b"generation-key"], [source])[0].status is KVStatus.OK
        destination = bytearray(len(source))
        assert (
            client.batch_get([b"generation-key"], [destination])[0].status
            is KVStatus.OK
        )
        assert destination == source
    finally:
        client.close()
        agent.close(timeout=5)


def test_lost_reserve_response_retries_same_request_without_new_extent(
    client_stack, monkeypatch
):
    client, service = client_stack
    original = service.batch_reserve
    lost_once = False

    def lose_first_response(items, **kwargs):
        nonlocal lost_once
        result = original(items, **kwargs)
        if not lost_once:
            lost_once = True
            raise TimeoutError("simulated lost metadata response")
        return result

    monkeypatch.setattr(service, "batch_reserve", lose_first_response)
    [stored] = client.batch_set(
        [SetItem(b"ambiguous", bytearray(b"payload"), request_id="stable-request")]
    )
    assert stored.status is KVStatus.OK
    assert service.get_metrics()["objects_ready"] == 1
    assert service.get_metrics()["idempotency_hits"] >= 1


def test_registration_failure_releases_reservation(client_stack, monkeypatch):
    client, service = client_stack

    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(client._agent, "register_memory", fail_registration)
    with pytest.raises(RuntimeError, match="registration failed"):
        client.batch_set([b"cleanup-write"], [bytearray(b"payload")])
    capacity = service.get_capacity(7)
    assert capacity.free_bytes == capacity.capacity_bytes
    assert capacity.reclaim_pending_bytes == 0
    service.validate_invariants()


def test_registration_failure_releases_read_lease(client_stack, monkeypatch):
    client, service = client_stack
    assert client.batch_set([b"cleanup-read"], [bytearray(b"payload")])[0].ok

    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(client._agent, "register_memory", fail_registration)
    with pytest.raises(RuntimeError, match="registration failed"):
        client.batch_get([b"cleanup-read"], [bytearray(7)])
    assert service.get_metrics()["leases_active"] == 0
    service.validate_invariants()


def test_metadata_batches_obey_byte_limit(tmp_path):
    service = ShardNamingService(ServiceConfig(auto_evict=False, max_batch_bytes=80))
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "byte-batches.bin",
            16 * 4096,
            logical_device_id=6,
            create=True,
            direct_io=False,
        )
    )
    service.register_device(
        DeviceRegistration(6, "inproc://byte-batches", agent.size, 4096, 1)
    )
    client = ShardClient(ClientConfig(service=service, agent=agent))
    try:
        keys = [bytes([97 + index]) * 50 for index in range(3)]
        assert client.batch_exists(keys) == [False, False, False]
        assert client.get_metrics()["metadata_calls"] == 3
    finally:
        client.close()
        agent.close(timeout=5)


def test_async_submission_has_bounded_admission(tmp_path, monkeypatch):
    service = ShardNamingService(ServiceConfig(auto_evict=False))
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "async-admission.bin",
            4096,
            logical_device_id=8,
            create=True,
            direct_io=False,
        )
    )
    service.register_device(
        DeviceRegistration(8, "inproc://async-admission", agent.size, 4096, 1)
    )
    client = ShardClient(
        ClientConfig(service=service, agent=agent, max_inflight_batches=1)
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked(items, sources, timeout):
        entered.set()
        assert release.wait(5)
        return [KVResult(b"first", KVStatus.OK)]

    monkeypatch.setattr(client, "batch_set", blocked)
    try:
        first = client.submit_batch_set([b"first"], [bytearray(b"1")])
        assert entered.wait(5)
        second = client.submit_batch_set([b"second"], [bytearray(b"2")])
        assert client.wait(second, timeout=1)[0].status is KVStatus.RESOURCE_EXHAUSTED
        release.set()
        assert client.wait(first, timeout=5)[0].status is KVStatus.OK
    finally:
        release.set()
        client.close()
        agent.close(timeout=5)


def test_bootstrap_configures_ucx_listener_and_peer_route(tmp_path):
    client = create_client(
        {
            "file_path": tmp_path / "bootstrap-ucx.bin",
            "size": 64 * 1024,
            "device_id": 11,
            "create": True,
            "direct_io": False,
            "embedded_service": True,
            "embedded_service_name": str(tmp_path),
            "ucx_listen_host": "127.0.0.1",
            "ucx_listen_port": 0,
            "remote_devices": [
                {
                    "device_id": 12,
                    "host": "127.0.0.1",
                    "port": 19100,
                    "transport": "ucx",
                }
            ],
        }
    )
    try:
        agent = client._agent
        endpoint = agent.ucx_endpoint
        assert endpoint is not None
        assert endpoint.port > 0
        assert agent._config.remote_devices[0].transport == "ucx"
        [registered] = client._config.service.list_devices()
        assert registered.agent_endpoint == f"ucx://127.0.0.1:{endpoint.port}"
    finally:
        client.close()
