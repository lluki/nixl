# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket
import threading
import time

import pytest
from nixl_shard.service import (
    CommitItem,
    DeviceRegistration,
    DeviceState,
    IoTerminalReport,
    IOTokenKind,
    LookupItem,
    ReserveItem,
    ResultCode,
    ServiceConfig,
    ShardNamingService,
)
from nixl_shard.service_rpc import ShardNamingTCPClient, ShardNamingTCPServer

PAGE = 4096


def _server_process(control):
    service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=1)
    with ShardNamingTCPServer(service) as server:
        control.send(server.address)
        control.recv()


def test_tcp_server_rejects_non_loopback_binding():
    service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=1)
    with pytest.raises(ValueError, match="loopback"):
        ShardNamingTCPServer(service, host="0.0.0.0")


def test_tcp_proxy_round_trip_across_process_and_restart_epoch():
    parent, child = multiprocessing.Pipe()
    process = multiprocessing.Process(target=_server_process, args=(child,))
    process.start()
    try:
        address = parent.recv()
        client = ShardNamingTCPClient(*address, timeout=2)
        registered = client.register_device(
            DeviceRegistration(3, "127.0.0.1:7003", 2 * PAGE, PAGE, 4)
        )
        assert registered.code is ResultCode.OK
        [reserved] = client.batch_reserve(
            [ReserveItem(b"rpc-key", 42, "rpc-client", "reserve")]
        )
        assert reserved.code is ResultCode.OK
        reservation = reserved.reservation
        [committed] = client.batch_commit(
            [
                CommitItem(
                    reservation.reservation_id,
                    reservation.generation,
                    reservation.service_epoch,
                    "rpc-client",
                    "commit",
                )
            ]
        )
        assert committed.code is ResultCode.OK
        assert isinstance(committed.mapping.locations, tuple)
        [lookup] = client.batch_lookup([LookupItem(b"rpc-key", "rpc-reader", "lookup")])
        assert lookup.code is ResultCode.OK
        assert lookup.mapping == committed.mapping
        assert client.batch_exists([b"rpc-key", b"missing"]) == [True, False]
        assert client.get_capacity(3).ready_bytes == PAGE
        assert client.validate_invariants() is None

        assert client.restart() == 2
        assert client.health()["service_epoch"] == 2
        [stale] = client.batch_commit(
            [
                CommitItem(
                    reservation.reservation_id,
                    reservation.generation,
                    reservation.service_epoch,
                    "rpc-client",
                    "commit-after-restart",
                )
            ]
        )
        assert stale.code is ResultCode.STALE_TOKEN
    finally:
        if process.is_alive():
            parent.send("stop")
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_tcp_proxy_enforces_rpc_timeout():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def stall():
        connection, _ = listener.accept()
        with connection:
            time.sleep(0.2)

    thread = threading.Thread(target=stall)
    thread.start()
    try:
        client = ShardNamingTCPClient(*listener.getsockname(), timeout=0.03)
        with pytest.raises(TimeoutError):
            client.health()
    finally:
        listener.close()
        thread.join(timeout=1)


def test_tcp_proxy_device_drain_and_generation_replacement():
    service = ShardNamingService(ServiceConfig(auto_evict=False), epoch=1)
    with ShardNamingTCPServer(service) as server:
        client = ShardNamingTCPClient(*server.address, timeout=2)
        registered = client.register_device(
            DeviceRegistration(
                3,
                "127.0.0.1:7003",
                2 * PAGE,
                PAGE,
                4,
                device_generation=4,
            )
        )
        assert registered.code is ResultCode.OK
        [reserved] = client.batch_reserve(
            [ReserveItem(b"draining", 42, "rpc-client", "reserve-draining")]
        )
        reservation = reserved.reservation

        draining = client.stop_device(3, 4, device_generation=4, grace_period_s=5)
        assert draining.code is ResultCode.OK
        assert draining.device.state is DeviceState.DRAINING
        assert draining.device.outstanding_reservations == 1
        [blocked] = client.batch_reserve(
            [ReserveItem(b"blocked", 42, "rpc-client", "reserve-blocked")]
        )
        assert blocked.code is ResultCode.UNAVAILABLE

        terminal = client.report_io_terminal(
            IoTerminalReport(
                reservation.reservation_id,
                IOTokenKind.WRITE,
                reservation.service_epoch,
                reservation.device_id,
                reservation.agent_epoch,
                reservation.generation,
                "rpc-client",
                "terminal-draining",
                device_generation=reservation.device_generation,
            )
        )
        assert terminal.code is ResultCode.OK
        status = client.get_device_status(3)
        assert status.device.state is DeviceState.OFFLINE
        assert status.device.outstanding_io == 0

        stale_generation = client.register_device(
            DeviceRegistration(
                3,
                "127.0.0.1:7004",
                2 * PAGE,
                PAGE,
                5,
                device_generation=4,
            )
        )
        assert stale_generation.code is ResultCode.STALE_TOKEN
        replacement = client.register_device(
            DeviceRegistration(
                3,
                "127.0.0.1:7004",
                2 * PAGE,
                PAGE,
                5,
                device_generation=5,
            )
        )
        assert replacement.code is ResultCode.OK
        assert replacement.device.state is DeviceState.ONLINE
