# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import socket
import threading
import time

import nixl_shard.agent as agent_module
import pytest
from nixl_shard import (
    AgentConfig,
    CompletionStatus,
    IoItem,
    RemoteDevice,
    ShardAgent,
    TcpEndpoint,
    buffer_region_from_writable,
)


def _server(tmp_path, **kwargs):
    return ShardAgent.open(
        AgentConfig(
            tmp_path / "server-device.bin",
            4096,
            logical_device_id=2,
            create=True,
            direct_io=False,
            tcp_listen_host="127.0.0.1",
            tcp_listen_port=0,
            **kwargs,
        )
    )


def _client(tmp_path, endpoint, remote_generation=1, **kwargs):
    return ShardAgent.open(
        AgentConfig(
            tmp_path / "client-device.bin",
            4096,
            logical_device_id=1,
            create=True,
            direct_io=False,
            remote_devices=(
                RemoteDevice(2, endpoint.host, endpoint.port, remote_generation),
            ),
            **kwargs,
        )
    )


def _server_process(file_path, control):
    try:
        with ShardAgent.open(
            AgentConfig(
                file_path,
                4096,
                logical_device_id=2,
                create=True,
                direct_io=False,
                tcp_listen_host="127.0.0.1",
            )
        ) as server:
            endpoint = server.tcp_endpoint
            assert endpoint is not None
            control.send(("ready", endpoint.host, endpoint.port))
            control.recv()
    except BaseException as exc:
        control.send(("error", repr(exc)))
        raise
    finally:
        control.close()


def test_mixed_local_and_remote_batch_round_trip(tmp_path):
    with _server(tmp_path) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint) as client:
            data = bytearray(256)
            data[:64] = b"L" * 64
            data[64:128] = b"R" * 64
            memory = client.register_memory([buffer_region_from_writable(data)])

            results = client.store_batch(
                [
                    IoItem(1, 128, memory, 0, 0, 64),
                    IoItem(2, 256, memory, 0, 64, 64),
                ],
                timeout=5,
            )
            assert [result.status for result in results] == [
                CompletionStatus.OK,
                CompletionStatus.OK,
            ]

            data[:128] = bytes(128)
            results = client.load_batch(
                [
                    IoItem(1, 128, memory, 0, 0, 64),
                    IoItem(2, 256, memory, 0, 64, 64),
                ],
                timeout=5,
            )
            assert [result.status for result in results] == [
                CompletionStatus.OK,
                CompletionStatus.OK,
            ]
            assert data[:64] == b"L" * 64
            assert data[64:128] == b"R" * 64
            client.unregister_memory(memory)

            metrics = client.get_metrics()
            assert metrics.submitted_ops == 4
            assert metrics.completed_ops == 4
            assert metrics.bytes_transferred == 256
            assert metrics.error_ops == 0


def test_remote_round_trip_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent_control, child_control = context.Pipe()
    process = context.Process(
        target=_server_process,
        args=(str(tmp_path / "process-server.bin"), child_control),
    )
    process.start()
    message = parent_control.recv()
    assert message[0] == "ready", message
    endpoint = TcpEndpoint(message[1], message[2])
    try:
        with _client(tmp_path, endpoint) as client:
            data = bytearray(b"p" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            item = IoItem(2, 512, memory, 0, 0, 64)
            assert (
                client.store_batch([item], timeout=5)[0].status is CompletionStatus.OK
            )
            data[:] = bytes(64)
            assert client.load_batch([item], timeout=5)[0].status is CompletionStatus.OK
            assert data == b"p" * 64
            client.unregister_memory(memory)
    finally:
        parent_control.send("stop")
        parent_control.close()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0


def test_remote_direct_io_uses_aligned_staging(tmp_path):
    with ShardAgent.open(
        AgentConfig(
            tmp_path / "direct-server-device.bin",
            8192,
            logical_device_id=2,
            create=True,
            direct_io=True,
            tcp_listen_host="127.0.0.1",
        )
    ) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint) as client:
            data = bytearray(b"d" * 4096)
            memory = client.register_memory([buffer_region_from_writable(data)])
            item = IoItem(2, 4096, memory, 0, 0, 4096)
            assert (
                client.store_batch([item], timeout=5)[0].status is CompletionStatus.OK
            )
            data[:] = bytes(4096)
            assert client.load_batch([item], timeout=5)[0].status is CompletionStatus.OK
            assert data == b"d" * 4096
            client.unregister_memory(memory)


def test_remote_per_item_error_does_not_fail_batch(tmp_path):
    with _server(tmp_path) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint) as client:
            data = bytearray(b"x" * 128)
            memory = client.register_memory([buffer_region_from_writable(data)])
            results = client.store_batch(
                [
                    IoItem(2, 0, memory, 0, 0, 64),
                    IoItem(2, 4090, memory, 0, 64, 64),
                ],
                timeout=5,
            )
            assert [result.status for result in results] == [
                CompletionStatus.OK,
                CompletionStatus.OUT_OF_RANGE,
            ]
            client.unregister_memory(memory)


def test_remote_rejects_stale_logical_device_generation(tmp_path):
    with _server(tmp_path, logical_device_generation=2) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint, remote_generation=1) as client:
            data = bytearray(b"x" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            [result] = client.store_batch(
                [IoItem(2, 0, memory, 0, 0, 64, device_generation=1)],
                timeout=5,
            )
            assert result.status is CompletionStatus.UNAVAILABLE
            assert "generation" in result.detail
            client.unregister_memory(memory)


def test_remote_rejects_stale_naming_authority_epoch(tmp_path):
    with _server(tmp_path, authority_epoch=2) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint) as client:
            data = bytearray(b"x" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            [result] = client.store_batch(
                [IoItem(2, 0, memory, 0, 0, 64, authority_epoch=1)],
                timeout=5,
            )
            assert result.status is CompletionStatus.UNAVAILABLE
            assert "authority epoch" in result.detail
            client.unregister_memory(memory)


def test_data_owning_agent_reports_terminal_io(tmp_path):
    reports = []
    with _server(tmp_path, io_terminal_callback=reports.append) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint) as client:
            data = bytearray(b"x" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            report = {"token_id": "reservation-1", "token_kind": "write"}
            [result] = client.store_batch(
                [
                    IoItem(
                        2,
                        0,
                        memory,
                        0,
                        0,
                        64,
                        terminal_report=report,
                    )
                ],
                timeout=5,
            )
            assert result.status is CompletionStatus.OK
            assert reports == [report]
            client.unregister_memory(memory)


def test_data_owning_agent_reports_fragmented_token_after_all_io(tmp_path, monkeypatch):
    writes = []
    reports = []

    def report_terminal(report):
        assert writes == [(0, 64), (64, 64)]
        reports.append(report)

    with _server(tmp_path, io_terminal_callback=report_terminal) as server:
        assert server.tcp_endpoint is not None
        original_pwrite = server._remote_pwrite

        def observed_pwrite(data, offset):
            result = original_pwrite(data, offset)
            writes.append((offset, len(data)))
            return result

        monkeypatch.setattr(server, "_remote_pwrite", observed_pwrite)
        with _client(tmp_path, server.tcp_endpoint) as client:
            data = bytearray(b"x" * 128)
            memory = client.register_memory([buffer_region_from_writable(data)])
            first_report = {
                "token_id": "reservation-1",
                "token_kind": "write",
                "request_id": "fragment-1",
            }
            second_report = {**first_report, "request_id": "fragment-2"}
            results = client.store_batch(
                [
                    IoItem(
                        2,
                        0,
                        memory,
                        0,
                        0,
                        64,
                        terminal_report=first_report,
                    ),
                    IoItem(
                        2,
                        64,
                        memory,
                        0,
                        64,
                        64,
                        terminal_report=second_report,
                    ),
                ],
                timeout=5,
            )
            assert [result.status for result in results] == [
                CompletionStatus.OK,
                CompletionStatus.OK,
            ]
            assert reports == [first_report]
            client.unregister_memory(memory)


def test_remote_staging_limit_applies_backpressure(tmp_path, monkeypatch):
    with _server(tmp_path) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint, tcp_max_staging_bytes=32) as client:
            data = bytearray(b"x" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            monkeypatch.setattr(
                agent_module.ctypes,
                "string_at",
                lambda *args: (_ for _ in ()).throw(
                    AssertionError("payload materialized before staging admission")
                ),
            )
            [result] = client.store_batch([IoItem(2, 0, memory, 0, 0, 64)], timeout=5)
            assert result.status is CompletionStatus.RESOURCE_EXHAUSTED
            assert "staging budget" in result.detail
            client.unregister_memory(memory)


def test_successful_remote_store_rejects_short_byte_count(tmp_path, monkeypatch):
    endpoint = TcpEndpoint("127.0.0.1", 1)
    with _client(tmp_path, endpoint) as client:
        monkeypatch.setattr(
            client._transport,
            "request",
            lambda *args, **kwargs: (
                [{"status": "OK", "bytes_transferred": 63, "detail": ""}],
                b"",
            ),
        )
        data = bytearray(b"x" * 64)
        memory = client.register_memory([buffer_region_from_writable(data)])
        [result] = client.store_batch([IoItem(2, 0, memory, 0, 0, 64)], timeout=5)
        assert result.status is CompletionStatus.DATA_LOSS
        assert "short byte count" in result.detail
        client.unregister_memory(memory)


def test_disconnect_is_reported_unavailable(tmp_path):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    endpoint = TcpEndpoint("127.0.0.1", port)

    with _client(tmp_path, endpoint, tcp_request_timeout=0.2) as client:
        data = bytearray(b"x" * 64)
        memory = client.register_memory([buffer_region_from_writable(data)])
        [result] = client.store_batch([IoItem(2, 0, memory, 0, 0, 64)], timeout=5)
        assert result.status is CompletionStatus.UNAVAILABLE
        client.unregister_memory(memory)


def test_remote_disconnect_after_send_is_terminal_and_releases_memory(tmp_path):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def stall():
        connection, _ = listener.accept()
        with connection:
            time.sleep(0.2)
        listener.close()

    thread = threading.Thread(target=stall)
    thread.start()
    endpoint = TcpEndpoint("127.0.0.1", port)
    with _client(tmp_path, endpoint, tcp_request_timeout=0.05) as client:
        data = bytearray(b"x" * 64)
        memory = client.register_memory([buffer_region_from_writable(data)])
        [result] = client.store_batch([IoItem(2, 0, memory, 0, 0, 64)], timeout=5)
        assert result.status is CompletionStatus.UNAVAILABLE
        client.unregister_memory(memory)
    thread.join(2)
    assert not thread.is_alive()


def test_post_send_delay_waits_for_terminal_reply(tmp_path, monkeypatch):
    with _server(tmp_path, tcp_request_timeout=0.05) as server:
        assert server.tcp_endpoint is not None
        original = server._tcp_server._handler

        def delayed_handler(*args, **kwargs):
            time.sleep(0.2)
            return original(*args, **kwargs)

        monkeypatch.setattr(server._tcp_server, "_handler", delayed_handler)
        with _client(tmp_path, server.tcp_endpoint, tcp_request_timeout=0.05) as client:
            data = bytearray(b"x" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            [result] = client.store_batch([IoItem(2, 0, memory, 0, 0, 64)], timeout=2)
            assert result.status is CompletionStatus.OK
            client.unregister_memory(memory)


def test_cancel_waits_for_remote_terminal_before_buffer_release(tmp_path, monkeypatch):
    with _server(tmp_path) as server:
        assert server.tcp_endpoint is not None
        with _client(tmp_path, server.tcp_endpoint) as client:
            entered = threading.Event()
            release = threading.Event()
            original = client._transport.request

            def delayed_request(*args, **kwargs):
                entered.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            monkeypatch.setattr(client._transport, "request", delayed_request)
            data = bytearray(b"x" * 64)
            memory = client.register_memory([buffer_region_from_writable(data)])
            [handle] = client.submit_batch_store([IoItem(2, 0, memory, 0, 0, 64)])
            assert entered.wait(5)
            client.cancel([handle])
            with pytest.raises(RuntimeError, match="active operation"):
                client.unregister_memory(memory)
            release.set()
            [result] = client.wait([handle], timeout=5)
            assert result.status is CompletionStatus.CANCELLED
            client.unregister_memory(memory)
            assert client.get_metrics().cancelled_ops == 1


def test_server_close_timeout_bounds_active_handler(tmp_path, monkeypatch):
    server = _server(tmp_path)
    assert server.tcp_endpoint is not None
    client = _client(tmp_path, server.tcp_endpoint)
    entered = threading.Event()
    release = threading.Event()
    original = server._tcp_server._handler

    def blocked_handler(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(server._tcp_server, "_handler", blocked_handler)
    data = bytearray(b"x" * 64)
    memory = client.register_memory([buffer_region_from_writable(data)])
    [handle] = client.submit_batch_store([IoItem(2, 0, memory, 0, 0, 64)])
    try:
        assert entered.wait(5)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="TCP server"):
            server.close(timeout=0.05)
        assert time.monotonic() - started < 1
        release.set()
        [result] = client.wait([handle], timeout=5)
        assert result.status is CompletionStatus.UNAVAILABLE
        client.unregister_memory(memory)
        server.close(timeout=5)
    finally:
        release.set()
        client.close(timeout=5)
        try:
            server.close(timeout=5)
        except TimeoutError:
            pass


def test_server_shutdown_bounds_incomplete_connections(tmp_path):
    server = _server(tmp_path, tcp_request_timeout=0.05)
    assert server.tcp_endpoint is not None
    connection = socket.create_connection(
        (server.tcp_endpoint.host, server.tcp_endpoint.port)
    )
    started = time.monotonic()
    server.close()
    elapsed = time.monotonic() - started
    connection.close()
    assert elapsed < 2
