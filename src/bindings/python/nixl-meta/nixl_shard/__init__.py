# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rack-scale NIXL storage primitives."""

from .agent import (
    AgentConfig,
    AgentMetrics,
    BufferRegion,
    Completion,
    CompletionStatus,
    IoItem,
    MemoryHandle,
    OperationHandle,
    OperationTimeoutError,
    ShardAgent,
    buffer_region_from_writable,
)
from .bootstrap import create_client
from .client import (
    ClientConfig,
    ClientOperationHandle,
    GetItem,
    KVResult,
    KVStatus,
    SetItem,
    ShardClient,
)
from .service import (
    AbortItem,
    CommitItem,
    DeviceInfo,
    DeviceManagementResult,
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
from .service_rpc import ShardNamingTCPClient, ShardNamingTCPServer
from .transport import RemoteDevice, TcpEndpoint

__all__ = [
    "AgentConfig",
    "AgentMetrics",
    "BufferRegion",
    "Completion",
    "CompletionStatus",
    "ClientConfig",
    "ClientOperationHandle",
    "DeviceRegistration",
    "DeviceInfo",
    "DeviceManagementResult",
    "DeviceState",
    "GetItem",
    "IoTerminalReport",
    "IOTokenKind",
    "IoItem",
    "KVResult",
    "KVStatus",
    "LookupItem",
    "MemoryHandle",
    "OperationHandle",
    "OperationTimeoutError",
    "RemoteDevice",
    "ReserveItem",
    "ResultCode",
    "ServiceConfig",
    "SetItem",
    "ShardAgent",
    "ShardClient",
    "ShardNamingService",
    "ShardNamingTCPClient",
    "ShardNamingTCPServer",
    "TcpEndpoint",
    "buffer_region_from_writable",
    "create_client",
    "AbortItem",
    "CommitItem",
    "ReleaseReadItem",
    "TouchItem",
]
