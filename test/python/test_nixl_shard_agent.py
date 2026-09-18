# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import random
import threading

import nixl_shard.agent as shard_agent_module
import pytest
from nixl_shard import (
    AgentConfig,
    BufferRegion,
    CompletionStatus,
    IoItem,
    OperationTimeoutError,
    ShardAgent,
    buffer_region_from_writable,
)

import nixl._utils as nixl_utils


def _fill(address: int, value: int, length: int) -> None:
    ctypes.memset(address, value, length)


def _bytes(address: int, length: int) -> bytes:
    return ctypes.string_at(address, length)


def test_create_reopen_and_reject_size_mismatch(tmp_path):
    path = tmp_path / "device.bin"
    config = AgentConfig(path, 8192, logical_device_id=7, create=True, direct_io=False)

    with ShardAgent.open(config):
        assert path.stat().st_size == 8192

    with ShardAgent.open(AgentConfig(path, 8192, logical_device_id=7, direct_io=False)):
        pass

    with pytest.raises(ValueError, match="does not match"):
        ShardAgent.open(AgentConfig(path, 4096, logical_device_id=7, direct_io=False))
    assert path.stat().st_size == 8192


def test_safe_release_capability_negotiates_with_updated_runtime(tmp_path):
    core_capability = getattr(
        shard_agent_module.nixl_bindings,
        "HAVE_SAFE_POSIX_ERROR_RELEASE_CORE",
        False,
    )
    if not core_capability:
        pytest.skip("installed NIXL runtime predates the safe-release capability")

    path = tmp_path / "device.bin"
    with ShardAgent.open(
        AgentConfig(path, 4096, logical_device_id=18, create=True, direct_io=False)
    ) as agent:
        backend_capability = (
            agent._nixl.get_backend_params("POSIX").get("safe_error_release") == "true"
        )
        assert agent._safe_posix_error_release == (
            core_capability and backend_capability
        )


def test_open_rejects_symlink_and_concurrent_owner(tmp_path):
    path = tmp_path / "device.bin"
    link = tmp_path / "device-link.bin"
    config = AgentConfig(path, 4096, logical_device_id=7, create=True, direct_io=False)
    with ShardAgent.open(config):
        with pytest.raises(BlockingIOError):
            ShardAgent.open(
                AgentConfig(path, 4096, logical_device_id=7, direct_io=False)
            )

    link.symlink_to(path)
    with pytest.raises(OSError):
        ShardAgent.open(AgentConfig(link, 4096, logical_device_id=7, direct_io=False))


def test_create_rolls_back_after_nixl_registration_failure(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"

    def fail_registration(*args, **kwargs):
        raise RuntimeError("injected registration failure")

    monkeypatch.setattr(
        shard_agent_module.nixl_agent, "register_memory", fail_registration
    )
    with pytest.raises(RuntimeError, match="injected"):
        ShardAgent.open(
            AgentConfig(path, 4096, logical_device_id=7, create=True, direct_io=False)
        )
    assert not path.exists()


def test_batch_local_posix_round_trip(tmp_path):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(8192)
    try:
        with ShardAgent.open(
            AgentConfig(path, 16384, logical_device_id=9, create=True, direct_io=False)
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 8192)])
            _fill(allocation, 0x2A, 4096)
            _fill(allocation + 4096, 0x6B, 4096)

            stores = agent.submit_batch_store(
                [
                    IoItem(9, 0, memory, 0, 0, 4096),
                    IoItem(9, 8192, memory, 0, 4096, 4096),
                ]
            )
            assert len(agent._pending) == 1
            store_results = agent.wait(stores, timeout=5)
            assert [result.status for result in store_results] == [
                CompletionStatus.OK,
                CompletionStatus.OK,
            ]

            _fill(allocation, 0, 8192)
            loads = agent.submit_batch_load(
                [
                    IoItem(9, 0, memory, 0, 0, 4096),
                    IoItem(9, 8192, memory, 0, 4096, 4096),
                ]
            )
            load_results = agent.wait(loads, timeout=5)
            assert [result.status for result in load_results] == [
                CompletionStatus.OK,
                CompletionStatus.OK,
            ]
            assert _bytes(allocation, 4096) == bytes([0x2A]) * 4096
            assert _bytes(allocation + 4096, 4096) == bytes([0x6B]) * 4096

            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_empty_batch_is_rejected(tmp_path):
    path = tmp_path / "device.bin"
    with ShardAgent.open(
        AgentConfig(path, 4096, logical_device_id=1, create=True, direct_io=False)
    ) as agent:
        with pytest.raises(ValueError, match="at least one"):
            agent.submit_batch_load([])


def test_writable_buffer_protocol_round_trip(tmp_path):
    path = tmp_path / "device.bin"
    payload = bytearray(b"nixlshard" * 8)
    expected = bytes(payload)
    with ShardAgent.open(
        AgentConfig(path, 4096, logical_device_id=8, create=True, direct_io=False)
    ) as agent:
        region = buffer_region_from_writable(payload)
        memory = agent.register_memory([region])
        item = IoItem(8, 128, memory, 0, 0, len(payload))
        assert agent.store_batch([item], timeout=5)[0].status is CompletionStatus.OK

        payload[:] = b"\0" * len(payload)
        assert agent.load_batch([item], timeout=5)[0].status is CompletionStatus.OK
        assert bytes(payload) == expected
        agent.unregister_memory(memory)


def test_inflight_limit_is_per_item_and_recovers(tmp_path):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(8192)
    try:
        with ShardAgent.open(
            AgentConfig(
                path,
                8192,
                logical_device_id=2,
                create=True,
                direct_io=False,
                max_inflight_ops=1,
            )
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 8192)])
            handles = agent.submit_batch_store(
                [
                    IoItem(2, 0, memory, 0, 0, 4096),
                    IoItem(2, 4096, memory, 0, 4096, 4096),
                ]
            )
            with pytest.raises(RuntimeError, match="active operation"):
                agent.unregister_memory(memory)
            results = agent.wait(handles, timeout=5)
            assert [result.status for result in results] == [
                CompletionStatus.OK,
                CompletionStatus.RESOURCE_EXHAUSTED,
            ]

            [retry] = agent.submit_batch_store([IoItem(2, 4096, memory, 0, 4096, 4096)])
            assert agent.wait([retry], timeout=5)[0].status is CompletionStatus.OK
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_submit_progresses_completed_request_before_capacity_check(
    tmp_path, monkeypatch
):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(8192)
    try:
        with ShardAgent.open(
            AgentConfig(
                path,
                8192,
                logical_device_id=16,
                create=True,
                direct_io=False,
                max_inflight_ops=1,
            )
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 8192)])
            monkeypatch.setattr(agent._nixl, "transfer", lambda handle: "PROC")
            monkeypatch.setattr(agent._nixl, "check_xfer_state", lambda handle: "PROC")
            [first] = agent.submit_batch_store([IoItem(16, 0, memory, 0, 0, 4096)])

            monkeypatch.setattr(agent._nixl, "check_xfer_state", lambda handle: "DONE")
            [second] = agent.submit_batch_store(
                [IoItem(16, 4096, memory, 0, 4096, 4096)]
            )
            results = agent.wait([first, second], timeout=5)
            assert [result.status for result in results] == [
                CompletionStatus.OK,
                CompletionStatus.OK,
            ]
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_blocking_wrapper_timeout_exposes_recoverable_handles(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(4096)
    try:
        with ShardAgent.open(
            AgentConfig(path, 4096, logical_device_id=17, create=True, direct_io=False)
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 4096)])
            monkeypatch.setattr(agent._nixl, "transfer", lambda handle: "PROC")
            monkeypatch.setattr(agent._nixl, "check_xfer_state", lambda handle: "PROC")

            with pytest.raises(OperationTimeoutError) as timeout:
                agent.store_batch([IoItem(17, 0, memory, 0, 0, 4096)], timeout=0.01)
            assert len(timeout.value.handles) == 1

            monkeypatch.setattr(agent._nixl, "check_xfer_state", lambda handle: "DONE")
            [result] = agent.wait(timeout.value.handles, timeout=5)
            assert result.status is CompletionStatus.OK
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_unregister_failure_keeps_registration_owned(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(4096)
    try:
        with ShardAgent.open(
            AgentConfig(path, 4096, logical_device_id=11, create=True, direct_io=False)
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 4096)])
            original = agent._nixl.deregister_memory

            def fail_deregister(*args, **kwargs):
                raise RuntimeError("injected deregistration failure")

            monkeypatch.setattr(agent._nixl, "deregister_memory", fail_deregister)
            with pytest.raises(RuntimeError, match="injected"):
                agent.unregister_memory(memory)
            assert memory.id in agent._memory

            monkeypatch.setattr(agent._nixl, "deregister_memory", original)
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_post_error_retries_release_before_terminal(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(4096)
    try:
        with ShardAgent.open(
            AgentConfig(path, 4096, logical_device_id=12, create=True, direct_io=False)
        ) as agent:
            agent._safe_posix_error_release = True
            memory = agent.register_memory([BufferRegion(allocation, 4096)])
            original_release = agent._nixl.release_xfer_handle
            release_calls = 0

            def fail_post(*args, **kwargs):
                raise RuntimeError("injected post failure")

            def release_after_retry(handle):
                nonlocal release_calls
                release_calls += 1
                if release_calls == 1:
                    raise RuntimeError("injected release busy")
                return original_release(handle)

            monkeypatch.setattr(agent._nixl, "transfer", fail_post)
            monkeypatch.setattr(agent._nixl, "release_xfer_handle", release_after_retry)
            [handle] = agent.submit_batch_store([IoItem(12, 0, memory, 0, 0, 4096)])
            [result] = agent.wait([handle], timeout=5)
            assert result.status is CompletionStatus.IO_ERROR
            assert release_calls == 2
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_status_error_releases_before_terminal(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(4096)
    try:
        with ShardAgent.open(
            AgentConfig(path, 4096, logical_device_id=13, create=True, direct_io=False)
        ) as agent:
            agent._safe_posix_error_release = True
            memory = agent.register_memory([BufferRegion(allocation, 4096)])

            monkeypatch.setattr(agent._nixl, "transfer", lambda handle: "PROC")

            def fail_status(*args, **kwargs):
                raise RuntimeError("injected status failure")

            monkeypatch.setattr(agent._nixl, "check_xfer_state", fail_status)
            [handle] = agent.submit_batch_load([IoItem(13, 0, memory, 0, 0, 4096)])
            [result] = agent.wait([handle], timeout=5)
            assert result.status is CompletionStatus.IO_ERROR
            assert "status failure" in result.detail
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_legacy_runtime_error_fails_closed_until_done(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(4096)
    try:
        with ShardAgent.open(
            AgentConfig(path, 4096, logical_device_id=15, create=True, direct_io=False)
        ) as agent:
            agent._safe_posix_error_release = False
            memory = agent.register_memory([BufferRegion(allocation, 4096)])
            monkeypatch.setattr(agent._nixl, "transfer", lambda handle: "PROC")

            def fail_status(*args, **kwargs):
                raise RuntimeError("injected legacy status failure")

            monkeypatch.setattr(agent._nixl, "check_xfer_state", fail_status)
            [handle] = agent.submit_batch_load([IoItem(15, 0, memory, 0, 0, 4096)])
            with pytest.raises(TimeoutError):
                agent.wait([handle], timeout=0.01)
            assert handle.id in agent._pending_handles

            monkeypatch.setattr(agent._nixl, "check_xfer_state", lambda handle: "DONE")
            [result] = agent.wait([handle], timeout=5)
            assert result.status is CompletionStatus.IO_ERROR
            assert "legacy status failure" in result.detail
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_concurrent_close_is_serialized_and_rejects_submit(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    agent = ShardAgent.open(
        AgentConfig(path, 4096, logical_device_id=14, create=True, direct_io=False)
    )
    entered = threading.Event()
    allow_close = threading.Event()
    errors = []
    original = agent._nixl.deregister_memory

    def slow_deregister(*args, **kwargs):
        entered.set()
        if not allow_close.wait(5):
            raise TimeoutError("test did not release close")
        return original(*args, **kwargs)

    monkeypatch.setattr(agent._nixl, "deregister_memory", slow_deregister)

    def run_close():
        try:
            agent.close(timeout=5)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=run_close)
    second = threading.Thread(target=run_close)
    submit_errors = []

    def run_submit():
        try:
            agent.submit_batch_load([])
        except Exception as exc:
            submit_errors.append(exc)

    submitter = threading.Thread(target=run_submit)
    first.start()
    assert entered.wait(5)
    second.start()
    submitter.start()
    allow_close.set()
    first.join(5)
    second.join(5)
    submitter.join(5)
    assert not first.is_alive()
    assert not second.is_alive()
    assert not submitter.is_alive()
    assert errors == []
    assert len(submit_errors) == 1
    assert isinstance(submit_errors[0], RuntimeError)
    assert "closed" in str(submit_errors[0])
    agent.close()


def test_close_failure_retains_retryable_ownership(tmp_path, monkeypatch):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(8192)
    agent = ShardAgent.open(
        AgentConfig(path, 8192, logical_device_id=15, create=True, direct_io=False)
    )
    original = agent._nixl.deregister_memory
    calls = 0
    try:
        agent.register_memory([BufferRegion(allocation, 4096)])
        agent.register_memory([BufferRegion(allocation + 4096, 4096)])

        def fail_second_deregister(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected close failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(agent._nixl, "deregister_memory", fail_second_deregister)
        with pytest.raises(RuntimeError, match="injected close failure"):
            agent.close()
        assert agent._state == "CLOSE_FAILED"
        assert len(agent._memory) == 1
        with pytest.raises(RuntimeError, match="close_failed"):
            agent.poll(0)

        monkeypatch.setattr(agent._nixl, "deregister_memory", original)
        agent.close()
        assert agent._state == "CLOSED"
    finally:
        monkeypatch.setattr(agent._nixl, "deregister_memory", original)
        if agent._state != "CLOSED":
            agent.close()
        nixl_utils.free_passthru(allocation)


def test_randomized_vector_round_trip(tmp_path):
    rng = random.Random(20260918)
    path = tmp_path / "device.bin"
    slot_size = 2048
    item_count = 24
    total_size = slot_size * item_count
    allocation = nixl_utils.malloc_passthru(total_size)
    try:
        with ShardAgent.open(
            AgentConfig(
                path,
                total_size,
                logical_device_id=10,
                create=True,
                direct_io=False,
            )
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, total_size)])
            slots = list(range(item_count))
            rng.shuffle(slots)
            items = []
            expected = {}
            for memory_slot, device_slot in enumerate(slots):
                length = rng.randint(1, slot_size - 64)
                file_offset = device_slot * slot_size + rng.randint(
                    0, slot_size - length
                )
                memory_offset = memory_slot * slot_size
                value = rng.randint(1, 255)
                _fill(allocation + memory_offset, value, length)
                items.append(IoItem(10, file_offset, memory, 0, memory_offset, length))
                expected[memory_offset] = bytes([value]) * length

            assert all(
                result.status is CompletionStatus.OK
                for result in agent.store_batch(items, timeout=5)
            )
            _fill(allocation, 0, total_size)
            assert all(
                result.status is CompletionStatus.OK
                for result in agent.load_batch(items, timeout=5)
            )
            for item in items:
                assert (
                    _bytes(allocation + item.memory_offset, item.length)
                    == expected[item.memory_offset]
                )
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)


def test_invalid_items_complete_per_item(tmp_path):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(4096)
    try:
        with ShardAgent.open(
            AgentConfig(path, 8192, logical_device_id=4, create=True, direct_io=False)
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 4096)])
            handles = agent.submit_batch_store(
                [
                    IoItem(5, 0, memory, 0, 0, 4096),
                    IoItem(4, 6144, memory, 0, 0, 4096),
                    IoItem(4, 0, memory, 0, 2048, 4096),
                ]
            )
            results = agent.wait(handles, timeout=1)
            assert [result.status for result in results] == [
                CompletionStatus.UNAVAILABLE,
                CompletionStatus.OUT_OF_RANGE,
                CompletionStatus.OUT_OF_RANGE,
            ]
    finally:
        nixl_utils.free_passthru(allocation)


def test_direct_io_rejects_unaligned_item(tmp_path):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(8192)
    try:
        with ShardAgent.open(
            AgentConfig(path, 8192, logical_device_id=3, create=True, direct_io=True)
        ) as agent:
            memory = agent.register_memory([BufferRegion(allocation, 8192)])
            handles = agent.submit_batch_store([IoItem(3, 1, memory, 0, 0, 4096)])
            [result] = agent.wait(handles, timeout=1)
            assert result.status is CompletionStatus.INVALID_ARGUMENT
            assert "unaligned" in result.detail
    finally:
        nixl_utils.free_passthru(allocation)


def test_direct_io_aligned_round_trip(tmp_path):
    path = tmp_path / "device.bin"
    allocation = nixl_utils.malloc_passthru(8192)
    aligned = (allocation + 4095) & ~4095
    try:
        with ShardAgent.open(
            AgentConfig(path, 8192, logical_device_id=6, create=True, direct_io=True)
        ) as agent:
            memory = agent.register_memory([BufferRegion(aligned, 4096)])
            _fill(aligned, 0x7C, 4096)
            [store] = agent.store_batch(
                [IoItem(6, 4096, memory, 0, 0, 4096)], timeout=5
            )
            assert store.status is CompletionStatus.OK
            assert store.terminal

            _fill(aligned, 0, 4096)
            [load] = agent.load_batch([IoItem(6, 4096, memory, 0, 0, 4096)], timeout=5)
            assert load.status is CompletionStatus.OK
            assert _bytes(aligned, 4096) == bytes([0x7C]) * 4096
            agent.unregister_memory(memory)
    finally:
        nixl_utils.free_passthru(allocation)
