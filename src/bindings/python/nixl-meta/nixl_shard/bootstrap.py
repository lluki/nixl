# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-only bootstrap for dynamic framework integrations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .agent import AgentConfig, ShardAgent
from .client import ClientConfig, ShardClient
from .service import (
    DeviceRegistration,
    IoTerminalReport,
    IOTokenKind,
    ResultCode,
    ServiceConfig,
    ShardNamingService,
)
from .service_rpc import ShardNamingTCPClient
from .transport import RemoteDevice

_EMBEDDED_SERVICES: dict[str, ShardNamingService] = {}


def _metadata_client(config: Mapping[str, Any]) -> Any:
    endpoint = config.get("metadata_endpoint")
    if endpoint:
        host, separator, port = str(endpoint).rpartition(":")
        if not separator or not host:
            raise ValueError("metadata_endpoint must use host:port syntax")
        return ShardNamingTCPClient(
            host,
            int(port),
            timeout=float(config.get("metadata_timeout_s", 5.0)),
            max_batch_items=int(config.get("max_batch_items", 256)),
            max_batch_bytes=int(config.get("max_batch_bytes", 1 << 20)),
        )
    if not config.get("embedded_service", False):
        raise ValueError("metadata_endpoint is required unless embedded_service=true")
    name = str(config.get("embedded_service_name", "default"))
    service = _EMBEDDED_SERVICES.get(name)
    if service is None:
        service = ShardNamingService(
            ServiceConfig(
                max_batch_items=int(config.get("max_batch_items", 256)),
                low_watermark=float(config.get("low_watermark", 0.60)),
                high_watermark=float(config.get("high_watermark", 0.80)),
                auto_evict=bool(config.get("auto_evict", True)),
            )
        )
        _EMBEDDED_SERVICES[name] = service
    return service


def create_client(config: Mapping[str, Any]) -> ShardClient:
    """Create the in-process agent/client expected by SGLang's dynamic backend.

    The embedded metadata option is intended for single-process functional
    validation. Rack deployments provide ``metadata_endpoint`` and explicit
    peer ``remote_devices`` instead.
    """

    required = ("file_path", "size", "device_id")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing NIXLShard client config fields: {missing}")
    file_path = os.fspath(config["file_path"])
    size = int(config["size"])
    device_id = int(config["device_id"])
    logical_device_generation = int(
        config.get("logical_device_generation", config.get("agent_epoch", 1))
    )
    alignment = int(config.get("alignment", 4096))
    remote_devices = tuple(
        RemoteDevice(
            int(remote["device_id"]),
            str(remote["host"]),
            int(remote["port"]),
            int(
                remote.get(
                    "logical_device_generation",
                    remote.get("agent_epoch", 1),
                )
            ),
            transport=str(remote.get("transport", "tcp")),
        )
        for remote in config.get("remote_devices", ())
    )
    create = bool(config.get("create", not os.path.exists(file_path)))
    service = _metadata_client(config)
    authority_epoch = getattr(service, "epoch", None)
    if authority_epoch is None:
        authority_epoch = int(service.health()["service_epoch"])

    def report_io_terminal(values: dict[str, Any]) -> None:
        values = dict(values)
        values["token_kind"] = IOTokenKind(values["token_kind"])
        result = service.report_io_terminal(IoTerminalReport(**values))
        if result.code not in (ResultCode.OK, ResultCode.STALE_TOKEN):
            raise RuntimeError(
                f"terminal I/O report failed: {result.code.value}: {result.detail}"
            )

    agent = ShardAgent.open(
        AgentConfig(
            file_path=file_path,
            size=size,
            logical_device_id=device_id,
            logical_device_generation=logical_device_generation,
            authority_epoch=int(authority_epoch),
            io_terminal_callback=report_io_terminal,
            create=create,
            direct_io=bool(config.get("direct_io", True)),
            alignment=alignment,
            max_inflight_ops=int(config.get("max_inflight_ops", 128)),
            remote_devices=remote_devices,
            tcp_listen_host=config.get("tcp_listen_host"),
            tcp_listen_port=int(config.get("tcp_listen_port", 0)),
            tcp_request_timeout=float(config.get("tcp_request_timeout", 5.0)),
            tcp_max_request_bytes=int(
                config.get("tcp_max_request_bytes", 64 * 1024 * 1024)
            ),
            tcp_max_staging_bytes=int(
                config.get("tcp_max_staging_bytes", 256 * 1024 * 1024)
            ),
            tcp_max_workers=int(config.get("tcp_max_workers", 8)),
            ucx_listen_host=config.get("ucx_listen_host"),
            ucx_listen_port=int(config.get("ucx_listen_port", 0)),
        )
    )
    try:
        ucx_endpoint = agent.ucx_endpoint
        tcp_endpoint = agent.tcp_endpoint
        if ucx_endpoint is not None:
            agent_endpoint = f"ucx://{ucx_endpoint.host}:{ucx_endpoint.port}"
        elif tcp_endpoint is not None:
            agent_endpoint = f"tcp://{tcp_endpoint.host}:{tcp_endpoint.port}"
        else:
            agent_endpoint = f"inproc://{device_id}"
        registration = service.register_device(
            DeviceRegistration(
                device_id=device_id,
                agent_endpoint=agent_endpoint,
                capacity_bytes=size,
                allocation_alignment=alignment,
                agent_epoch=logical_device_generation,
                numa_node=int(config.get("numa_node", -1)),
                failure_domain=config.get("failure_domain"),
                device_generation=logical_device_generation,
            )
        )
        if registration.code is not ResultCode.OK:
            raise RuntimeError(
                f"NIXLShard device registration failed: {registration.code.value}: "
                f"{registration.detail}"
            )
        return ShardClient(
            ClientConfig(
                service=service,
                agent=agent,
                client_id=str(config.get("client_id", f"device-{device_id}")),
                default_timeout_s=float(config.get("default_timeout_s", 30.0)),
                reservation_ttl_s=float(config.get("reservation_ttl_s", 30.0)),
                read_lease_ttl_s=float(config.get("read_lease_ttl_s", 30.0)),
                max_inflight_batches=int(config.get("max_inflight_batches", 16)),
                lazy_release=bool(config.get("lazy_release", False)),
                max_pending_lazy_releases=int(
                    config.get("max_pending_lazy_releases", 16)
                ),
                close_agent_on_close=True,
                heartbeat_device_id=device_id,
                heartbeat_agent_epoch=logical_device_generation,
                heartbeat_device_generation=logical_device_generation,
                heartbeat_interval_s=float(config.get("heartbeat_interval_s", 10.0)),
            )
        )
    except BaseException:
        agent.close(timeout=float(config.get("default_timeout_s", 30.0)))
        raise
