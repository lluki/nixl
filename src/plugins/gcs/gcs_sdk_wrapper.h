/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NIXL_SRC_PLUGINS_GCS_GCS_SDK_WRAPPER_H
#define NIXL_SRC_PLUGINS_GCS_GCS_SDK_WRAPPER_H

#include <cstddef>
#include <cstdint>

struct nixl_gcs_client;

extern "C" nixl_gcs_client *
nixl_gcs_client_create();
extern "C" void
nixl_gcs_client_destroy(nixl_gcs_client *);
extern "C" bool
nixl_gcs_put(nixl_gcs_client *,
             const char *bucket,
             const char *object,
             const void *data,
             std::size_t length,
             std::uint64_t offset);
extern "C" bool
nixl_gcs_get(nixl_gcs_client *,
             const char *bucket,
             const char *object,
             void *data,
             std::size_t length,
             std::uint64_t offset);
// 1 = exists, 0 = missing, -1 = request error.
extern "C" int
nixl_gcs_exists(nixl_gcs_client *, const char *bucket, const char *object);

#endif // NIXL_SRC_PLUGINS_GCS_GCS_SDK_WRAPPER_H
