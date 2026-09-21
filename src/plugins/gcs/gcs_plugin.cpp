/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gcs_backend.h"
#include "backend/backend_plugin.h"

using gcs_plugin_t = nixlBackendPluginCreator<nixlGcsEngine>;

namespace {
const nixl_b_params_t gcs_plugin_params = {{"bucket", ""}, {"num_threads", ""}};
}

#ifdef STATIC_PLUGIN_GCS
nixlBackendPlugin *
createStaticGCSPlugin() {
    return gcs_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "GCS", "0.1.0", gcs_plugin_params, {DRAM_SEG, OBJ_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return gcs_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "GCS", "0.1.0", gcs_plugin_params, {DRAM_SEG, OBJ_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
