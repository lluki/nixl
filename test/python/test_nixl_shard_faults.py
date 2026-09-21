# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import socket
import threading
import time

import pytest
from nixl_shard import (
    AgentConfig,
    ClientConfig,
    CommitItem,
    DeviceRegistration,
    GetItem,
    KVStatus,
    LookupItem,
    RemoteDevice,
    ReserveItem,
    ResultCode,
    ServiceConfig,
    SetItem,
    ShardAgent,
    ShardClient,
    ShardNamingService,
    ShardNamingTCPClient,
    ShardNamingTCPServer,
)

PAGE = 4096


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _naming_server_process(epoch: int, port: int, control) -> None:
    try:
        service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=epoch)
        with ShardNamingTCPServer(service, port=port) as server:
            control.send(("ready", *server.address))
            control.recv()
    except BaseException as exc:
        try:
            control.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        control.close()


def _remote_agent_process(path: str, control) -> None:
    try:
        with ShardAgent.open(
            AgentConfig(
                path,
                4 * PAGE,
                logical_device_id=2,
                logical_device_generation=1,
                create=True,
                direct_io=False,
                tcp_listen_host="127.0.0.1",
            )
        ) as agent:
            endpoint = agent.tcp_endpoint
            assert endpoint is not None
            control.send(("ready", endpoint.host, endpoint.port))
            control.recv()
    except BaseException as exc:
        try:
            control.send(("error", repr(exc)))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        control.close()


def _start_process(context, target, args):
    parent, child = context.Pipe()
    process = context.Process(target=target, args=(*args, child))
    process.start()
    child.close()
    assert parent.poll(10), "child process did not become ready"
    message = parent.recv()
    assert message[0] == "ready", message
    return process, parent, message[1:]


def _stop_process(process, control) -> None:
    if process.is_alive():
        control.send("stop")
    control.close()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)


def test_naming_process_restart_forgets_data_and_rejects_old_tokens():
    context = multiprocessing.get_context("spawn")
    first, first_control, address = _start_process(
        context, _naming_server_process, (1, 0)
    )
    host, port = address
    client = ShardNamingTCPClient(host, port, timeout=2)
    client.register_device(
        DeviceRegistration(7, "inproc://device-1", 2 * PAGE, PAGE, 1)
    )
    [ready_reserve] = client.batch_reserve(
        [ReserveItem(b"ready-before-crash", 1, "writer", "reserve-ready")]
    )
    ready_token = ready_reserve.reservation
    [ready_commit] = client.batch_commit(
        [
            CommitItem(
                ready_token.reservation_id,
                ready_token.generation,
                ready_token.service_epoch,
                "writer",
                "commit-ready",
            )
        ]
    )
    assert ready_commit.code is ResultCode.OK
    [pending] = client.batch_reserve(
        [ReserveItem(b"pending-at-crash", 1, "writer", "reserve-pending")]
    )
    old_token = pending.reservation

    first.terminate()
    first.join(timeout=5)
    first_control.close()
    assert not first.is_alive()

    second, second_control, replacement_address = _start_process(
        context, _naming_server_process, (2, port)
    )
    assert replacement_address == (host, port)
    replacement = ShardNamingTCPClient(host, port, timeout=2)
    try:
        registered = replacement.register_device(
            DeviceRegistration(
                7,
                "inproc://device-2",
                2 * PAGE,
                PAGE,
                2,
                device_generation=2,
            )
        )
        assert registered.code is ResultCode.OK
        [old_lookup] = replacement.batch_lookup(
            [LookupItem(b"ready-before-crash", "reader", "lookup-old")]
        )
        assert old_lookup.code is ResultCode.NOT_FOUND
        [stale_commit] = replacement.batch_commit(
            [
                CommitItem(
                    old_token.reservation_id,
                    old_token.generation,
                    old_token.service_epoch,
                    "writer",
                    "commit-after-process-restart",
                )
            ]
        )
        assert stale_commit.code is ResultCode.STALE_TOKEN
        assert replacement.health()["service_epoch"] == 2
    finally:
        _stop_process(second, second_control)
    assert second.exitcode == 0


def test_fresh_naming_authority_cannot_write_through_surviving_old_agent(
    tmp_path,
):
    old_authority_agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "old-authority-agent.bin",
            PAGE,
            logical_device_id=7,
            logical_device_generation=1,
            authority_epoch=1,
            create=True,
            direct_io=False,
        )
    )
    fresh_service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=2)
    registered = fresh_service.register_device(
        DeviceRegistration(
            7,
            "inproc://surviving-old-agent",
            PAGE,
            PAGE,
            agent_epoch=1,
            device_generation=1,
        )
    )
    assert registered.code is ResultCode.OK
    client = ShardClient(ClientConfig(service=fresh_service, agent=old_authority_agent))
    try:
        [result] = client.batch_set(
            [SetItem(b"must-be-fenced", bytearray(b"x"), preferred_device_id=7)]
        )
        assert result.status is KVStatus.UNAVAILABLE
        [lookup] = fresh_service.batch_lookup(
            [LookupItem(b"must-be-fenced", "reader", "lookup-fenced")]
        )
        assert lookup.code is ResultCode.NOT_FOUND
        fresh_service.validate_invariants()
    finally:
        client.close()
        old_authority_agent.close(timeout=5)


def test_expired_reservation_is_not_reused_while_data_io_is_delayed(
    tmp_path, monkeypatch
):
    clock = FakeClock()
    service = ShardNamingService(ServiceConfig(clock=clock, auto_evict=False), epoch=1)
    registered = service.register_device(
        DeviceRegistration(7, "inproc://device", PAGE, PAGE, 1)
    )
    assert registered.code is ResultCode.OK
    agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "delayed.bin",
            PAGE,
            logical_device_id=7,
            create=True,
            direct_io=False,
        )
    )
    client = ShardClient(
        ClientConfig(service=service, agent=agent, reservation_ttl_s=0.05)
    )
    entered = threading.Event()
    release = threading.Event()
    original_submit = agent.submit_batch_store
    outcome = {}

    def delayed_submit(items):
        entered.set()
        assert release.wait(5)
        return original_submit(items)

    def write() -> None:
        try:
            outcome["result"] = client.batch_set(
                [SetItem(b"delayed", bytearray(b"x" * PAGE), preferred_device_id=7)]
            )[0]
        except BaseException as exc:
            outcome["error"] = exc

    monkeypatch.setattr(agent, "submit_batch_store", delayed_submit)
    worker = threading.Thread(target=write)
    worker.start()
    try:
        assert entered.wait(5)
        clock.advance(1)
        assert service.reap_expired().code is ResultCode.OK
        capacity = service.get_capacity(7)
        assert capacity.free_bytes == 0
        assert capacity.reclaim_pending_bytes == PAGE
        [blocked] = service.batch_reserve(
            [
                ReserveItem(
                    b"must-not-reuse",
                    1,
                    "other-writer",
                    "reserve-while-io-active",
                    preferred_device_id=7,
                )
            ]
        )
        assert blocked.code is ResultCode.NO_SPACE
    finally:
        release.set()
        worker.join(timeout=10)
        client.close()
        agent.close(timeout=5)

    assert not worker.is_alive()
    assert "error" not in outcome
    assert outcome["result"].status is not KVStatus.OK
    capacity = service.get_capacity(7)
    assert capacity.free_bytes == PAGE
    assert capacity.reclaim_pending_bytes == 0
    [miss] = service.batch_lookup(
        [LookupItem(b"delayed", "reader", "lookup-expired-write")]
    )
    assert miss.code is ResultCode.NOT_FOUND
    service.validate_invariants()


def test_remote_agent_process_death_is_a_miss_and_local_io_stays_healthy(tmp_path):
    context = multiprocessing.get_context("spawn")
    process, control, endpoint = _start_process(
        context,
        _remote_agent_process,
        (str(tmp_path / "remote-agent.bin"),),
    )
    host, port = endpoint
    service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=1)
    service.register_device(DeviceRegistration(1, "inproc://local", 4 * PAGE, PAGE, 1))
    service.register_device(
        DeviceRegistration(2, f"tcp://{host}:{port}", 4 * PAGE, PAGE, 1)
    )
    local_agent = ShardAgent.open(
        AgentConfig(
            tmp_path / "local-agent.bin",
            4 * PAGE,
            logical_device_id=1,
            create=True,
            direct_io=False,
            remote_devices=(RemoteDevice(2, host, port, 1),),
            tcp_request_timeout=0.2,
        )
    )
    client = ShardClient(ClientConfig(service=service, agent=local_agent))
    try:
        remote_value = bytearray(b"remote" * 100)
        local_value = bytearray(b"local" * 100)
        [remote_set, local_set] = client.batch_set(
            [
                SetItem(b"remote", remote_value, preferred_device_id=2),
                SetItem(b"local", local_value, preferred_device_id=1),
            ]
        )
        assert remote_set.status is KVStatus.OK
        assert local_set.status is KVStatus.OK

        process.terminate()
        process.join(timeout=5)
        control.close()
        assert not process.is_alive()

        remote_destination = bytearray(len(remote_value))
        [remote_get] = client.batch_get([GetItem(b"remote", remote_destination)])
        assert remote_get.status is KVStatus.MISS

        local_destination = bytearray(len(local_value))
        [local_get] = client.batch_get([GetItem(b"local", local_destination)])
        assert local_get.status is KVStatus.OK
        assert local_destination == local_value
        service.validate_invariants()
    finally:
        client.close()
        local_agent.close(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if not control.closed:
            control.close()


def test_naming_rpc_bounds_incomplete_connections_and_recovers():
    service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=1)
    with ShardNamingTCPServer(service, request_timeout=1.0, max_workers=1) as server:
        blocker = socket.create_connection(server.address, timeout=1)
        blocker.sendall(b"\x00")
        deadline = time.monotonic() + 2
        while server._server._worker_slots._value != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server._server._worker_slots._value == 0

        client = ShardNamingTCPClient(*server.address, timeout=0.2)
        with pytest.raises((ConnectionError, OSError, TimeoutError)):
            client.health()

        blocker.close()
        deadline = time.monotonic() + 2
        while server._server._worker_slots._value != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server._server._worker_slots._value == 1
        assert client.health()["service_epoch"] == 1
