# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing

from nixl_shard import (
    AgentConfig,
    ClientConfig,
    DeviceRegistration,
    GetItem,
    KVStatus,
    RemoteDevice,
    ServiceConfig,
    SetItem,
    ShardAgent,
    ShardClient,
    ShardNamingService,
    ShardNamingTCPClient,
    ShardNamingTCPServer,
)

PAGE = 4096


def _remote_agent_process(path: str, control) -> None:
    try:
        with ShardAgent.open(
            AgentConfig(
                path,
                16 * PAGE,
                logical_device_id=2,
                create=True,
                direct_io=False,
                tcp_listen_host="127.0.0.1",
            )
        ) as agent:
            endpoint = agent.tcp_endpoint
            control.send((endpoint.host, endpoint.port))
            control.recv()
    finally:
        control.close()


def test_tcp_control_and_remote_data_end_to_end(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_remote_agent_process,
        args=(str(tmp_path / "remote.bin"), child),
    )
    process.start()
    host, port = parent.recv()

    service = ShardNamingService(ServiceConfig(auto_evict=False))
    naming_server = ShardNamingTCPServer(service).start()
    naming = ShardNamingTCPClient(*naming_server.address, timeout=2)
    local_agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "local.bin",
            16 * PAGE,
            logical_device_id=1,
            create=True,
            direct_io=False,
            remote_devices=(RemoteDevice(2, host, port),),
        )
    )
    client = ShardClient(ClientConfig(service=naming, agent=local_agent))
    try:
        naming.register_device(
            DeviceRegistration(1, "inproc://local", 16 * PAGE, PAGE, 1, numa_node=0)
        )
        naming.register_device(
            DeviceRegistration(
                2, f"tcp://{host}:{port}", 16 * PAGE, PAGE, 1, numa_node=1
            )
        )
        source = bytearray(b"remote-page" * 97)
        [stored] = client.batch_set(
            [SetItem(b"remote-key", source, preferred_device_id=2)]
        )
        assert stored.status is KVStatus.OK
        assert stored.device_id == 2

        destination = bytearray(len(source))
        [loaded] = client.batch_get([GetItem(b"remote-key", destination)])
        assert loaded.status is KVStatus.OK
        assert loaded.device_id == 2
        assert destination == source
        naming.validate_invariants()

        old_epoch = naming.health()["service_epoch"]
        assert naming.restart() != old_epoch
        [after_restart] = client.batch_get(
            [GetItem(b"remote-key", bytearray(len(source)))]
        )
        assert after_restart.status is KVStatus.MISS
    finally:
        client.close()
        local_agent.close(timeout=5)
        naming_server.close()
        if process.is_alive():
            parent.send("stop")
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_tiny_device_long_churn_evicts_and_reuses_safely(tmp_path):
    service = ShardNamingService(
        ServiceConfig(low_watermark=0.50, high_watermark=0.75, auto_evict=True)
    )
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "tiny.bin",
            8 * PAGE,
            logical_device_id=9,
            create=True,
            direct_io=False,
        )
    )
    service.register_device(DeviceRegistration(9, "inproc://tiny", 8 * PAGE, PAGE, 1))
    client = ShardClient(ClientConfig(service=service, agent=agent))
    expected = {}
    try:
        for generation in range(100):
            key = f"page-{generation}".encode()
            value = bytearray(bytes([generation % 251]) * 513)
            [result] = client.batch_set([SetItem(key, value, preferred_device_id=9)])
            assert result.status is KVStatus.OK
            expected[key] = value
            service.validate_invariants()

        present = client.batch_exists(list(expected))
        present_keys = [
            key for key, found in zip(expected, present, strict=True) if found
        ]
        assert present_keys
        assert len(present_keys) <= 6
        assert b"page-99" in present_keys
        destinations = [bytearray(len(expected[key])) for key in present_keys]
        loaded = client.batch_get(present_keys, destinations)
        assert all(result.status is KVStatus.OK for result in loaded)
        assert destinations == [expected[key] for key in present_keys]
        metrics = service.get_metrics()
        assert metrics["evictions"] > 0
        assert metrics["reclaimed_extents"] > 0
        service.validate_invariants()
    finally:
        client.close()
        agent.close(timeout=5)
