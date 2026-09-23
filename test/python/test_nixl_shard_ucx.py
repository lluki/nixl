# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused UCX staging and repeated-metadata regression tests."""

import ctypes
import json
import socket

import pytest

from nixl_shard import (
    AgentConfig,
    CompletionStatus,
    IoItem,
    RemoteDevice,
    ShardAgent,
    buffer_region_from_writable,
)
from nixl_shard import ucx_transport


def test_ucx_repeated_multi_item_round_trip(tmp_path, monkeypatch):
    trace_path = tmp_path / "ucx-trace.jsonl"
    monkeypatch.setenv("NIXLSHARD_TRACE_PATH", str(trace_path))
    server_config = AgentConfig(
        tmp_path / "ucx-server.bin",
        65536,
        logical_device_id=2,
        create=True,
        direct_io=True,
        ucx_listen_host="127.0.0.1",
    )
    try:
        server = ShardAgent.open(server_config)
    except RuntimeError as exc:
        if "UCX plugin is not available" in str(exc):
            pytest.skip(str(exc))
        raise
    with server:
        endpoint = server.ucx_endpoint
        assert endpoint is not None
        with ShardAgent.open(
            AgentConfig(
                tmp_path / "ucx-client.bin",
                65536,
                logical_device_id=1,
                create=True,
                direct_io=False,
                remote_devices=(
                    RemoteDevice(2, endpoint.host, endpoint.port, transport="ucx"),
                ),
            )
        ) as client:
            opened = 0
            create_connection = ucx_transport.socket.create_connection

            def counted_connection(*args, **kwargs):
                nonlocal opened
                opened += 1
                return create_connection(*args, **kwargs)

            monkeypatch.setattr(
                ucx_transport.socket, "create_connection", counted_connection
            )
            source = bytearray(b"a" * 4096 + b"b" * 4096)
            target = bytearray(8192)
            src = client.register_memory([buffer_region_from_writable(source)])
            dst = client.register_memory([buffer_region_from_writable(target)])
            for iteration in range(8):
                stores = [IoItem(2, i * 4096, src, 0, i * 4096, 4096) for i in range(2)]
                loads = [
                    IoItem(
                        2,
                        i * 4096,
                        dst,
                        0,
                        i * 4096,
                        4096,
                        request_id=f"load-{iteration}-{i}",
                    )
                    for i in range(2)
                ]
                assert all(
                    x.status is CompletionStatus.OK
                    for x in client.store_batch(stores, timeout=10)
                )
                assert all(
                    x.status is CompletionStatus.OK
                    for x in client.load_batch(loads, timeout=10)
                )
                assert target == source
            large_source = bytearray(b"c" * 16384)
            large_target = bytearray(16384)
            large_src = client.register_memory(
                [buffer_region_from_writable(large_source)]
            )
            large_dst = client.register_memory(
                [buffer_region_from_writable(large_target)]
            )
            large_store = IoItem(2, 16384, large_src, 0, 0, 16384)
            large_load = IoItem(2, 16384, large_dst, 0, 0, 16384)
            assert (
                client.store_batch([large_store], timeout=10)[0].status
                is CompletionStatus.OK
            )
            assert (
                client.load_batch([large_load], timeout=10)[0].status
                is CompletionStatus.OK
            )
            assert large_target == large_source
            assert all(
                x.status is CompletionStatus.OK
                for x in client.store_batch(stores, timeout=10)
            )
            assert client.ucx_transfer_bytes == 8 * 4 * 4096 + 2 * 16384 + 2 * 4096
            assert client._ucx_transport._pool._allocated == 8192 + 16384
            assert server._ucx_server._pool._allocated == 8192 + 16384
            assert opened == 1  # Nineteen batches used one control connection.
            trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
            loads = [event for event in trace if event["event"] == "ucx_load"]
            assert len(loads) == 9
            assert loads[0]["request_ids"] == ["load-0-0", "load-0-1"]
            assert loads[0]["server_stage_ns"] > 0
            assert loads[0]["rdma_transfer_ns"] > 0
            client.unregister_memory(src)
            client.unregister_memory(dst)
            client.unregister_memory(large_src)
            client.unregister_memory(large_dst)


def test_ucx_server_close_drains_idle_persistent_control(tmp_path):
    try:
        server = ShardAgent.open(
            AgentConfig(
                tmp_path / "ucx-close-server.bin",
                32768,
                logical_device_id=2,
                create=True,
                direct_io=False,
                ucx_listen_host="127.0.0.1",
                tcp_max_workers=4,
            )
        )
    except RuntimeError as exc:
        if "UCX plugin is not available" in str(exc):
            pytest.skip(str(exc))
        raise
    endpoint = server.ucx_endpoint
    assert endpoint is not None
    client = ShardAgent.open(
        AgentConfig(
            tmp_path / "ucx-close-client.bin",
            4096,
            logical_device_id=1,
            create=True,
            direct_io=False,
            tcp_max_workers=4,
            remote_devices=(
                RemoteDevice(2, endpoint.host, endpoint.port, transport="ucx"),
            ),
        )
    )
    try:
        source = bytearray(b"x" * 32768)
        memory = client.register_memory([buffer_region_from_writable(source)])
        handles = [
            client.submit_batch_store([IoItem(2, i * 4096, memory, 0, i * 4096, 4096)])[
                0
            ]
            for i in range(8)
        ]
        assert all(
            result.status is CompletionStatus.OK
            for result in client.wait(handles, timeout=20)
        )
        assert 1 <= client._ucx_transport._controls._total <= 4
        server.close(timeout=2)
        client.unregister_memory(memory)
    finally:
        client.close(timeout=5)
        server.close(timeout=5)


def test_ucx_staging_deregister_error_retains_allocation(monkeypatch):
    allocation = ctypes.create_string_buffer(4096)
    freed = []
    monkeypatch.setattr(
        ucx_transport.nixl_utils,
        "malloc_passthru",
        lambda length: ctypes.addressof(allocation),
    )
    monkeypatch.setattr(ucx_transport.nixl_utils, "free_passthru", freed.append)

    class Agent:
        def register_memory(self, *args, **kwargs):
            return object()

        def deregister_memory(self, *args, **kwargs):
            raise RuntimeError("registration may still be active")

    staging = ucx_transport._Staging(Agent(), 4096)
    with pytest.raises(RuntimeError, match="still be active"):
        staging.close()
    assert not freed


def test_ucx_unavailable_peer_fails_closed(tmp_path):
    reports = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        unavailable_port = listener.getsockname()[1]
    try:
        client = ShardAgent.open(
            AgentConfig(
                tmp_path / "ucx-unavailable.bin",
                4096,
                logical_device_id=1,
                create=True,
                direct_io=False,
                tcp_request_timeout=0.5,
                io_terminal_callback=reports.append,
                remote_devices=(
                    RemoteDevice(2, "127.0.0.1", unavailable_port, transport="ucx"),
                ),
            )
        )
    except RuntimeError as exc:
        if "UCX plugin is not available" in str(exc):
            pytest.skip(str(exc))
        raise
    with client:
        source = bytearray(b"x" * 4096)
        memory = client.register_memory([buffer_region_from_writable(source)])
        [completion] = client.store_batch(
            [
                IoItem(
                    2,
                    0,
                    memory,
                    0,
                    0,
                    4096,
                    terminal_report={"token_id": "unreachable"},
                )
            ],
            timeout=5,
        )
        assert completion.status is CompletionStatus.UNAVAILABLE
        assert client.ucx_transfer_bytes == 0
        assert not reports
        client.unregister_memory(memory)


def test_ucx_progress_error_does_not_release_active_handle():
    released = []

    class Agent:
        def get_xfer_descs(self, *args, **kwargs):
            return object()

        def initialize_xfer(self, *args, **kwargs):
            return "handle"

        def transfer(self, handle):
            return "PROC"

        def check_xfer_state(self, handle):
            raise RuntimeError("progress uncertain")

        def release_xfer_handle(self, handle):
            released.append(handle)

    peer = ucx_transport._UcxPeer(Agent())
    with pytest.raises(ucx_transport.UcxOwnershipError):
        peer.transfer("WRITE", 1, 2, 4096, "peer")
    assert not released


def test_ucx_pool_close_keeps_failed_registration_and_closes_others():
    closed = []

    class Stage:
        def __init__(self, name, fail):
            self.name = name
            self.fail = fail

        def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("deregister failed")

    pool = ucx_transport._StagingPool(None, 8192)
    failed = Stage("failed", True)
    good = Stage("good", False)
    pool._free = [failed, good]
    before = len(ucx_transport._PROCESS_ORPHANS)
    try:
        with pytest.raises(RuntimeError, match="deregister failed"):
            pool.close()
        assert closed == ["failed", "good"]
        assert pool._orphaned == [failed]
        assert pool._free == []
    finally:
        del ucx_transport._PROCESS_ORPHANS[before:]
