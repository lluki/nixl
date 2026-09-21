# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gcs_sources = [
    'gcs_backend.cpp',
    'gcs_backend.h',
    'gcs_plugin.cpp',
    'gcs_sdk_wrapper.h',
]

gcs_storage = dependency(
    'google_cloud_cpp_storage',
    method: 'cmake',
    modules: ['google-cloud-cpp::storage'],
    required: false)

if not gcs_storage.found()
    if is_explicit_enable or 'GCS' in static_plugins
        error('GCS plugin requested but google-cloud-cpp storage dependency not found')
    endif
    warning('google-cloud-cpp storage dependency not found, skipping GCS plugin build')
    subdir_done()
endif

if 'GCS' in static_plugins
    gcs_sdk_wrapper = static_library(
        'nixl_gcs_sdk_wrapper',
        'gcs_sdk_wrapper.cpp',
        dependencies: [gcs_storage],
        install: false)
else
    gcs_sdk_wrapper = shared_library(
        'nixl_gcs_sdk_wrapper',
        'gcs_sdk_wrapper.cpp',
        dependencies: [gcs_storage],
        install: true,
        install_dir: plugin_install_dir)
endif

gcs_deps = [nixl_infra, nixl_common_dep, dependency('asio', required: true)]
compile_defs = []
compile_flags = []

if 'GCS' in static_plugins
    gcs_backend_lib = static_library(
        'GCS', gcs_sources,
        dependencies: gcs_deps,
        link_with: [nixl_common_lib, gcs_sdk_wrapper],
        cpp_args: compile_defs + compile_flags,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')
else
    gcs_backend_lib = shared_library(
        'GCS', gcs_sources,
        dependencies: gcs_deps,
        link_with: [nixl_common_lib, gcs_sdk_wrapper],
        cpp_args: compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN:$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "GCS=' + gcs_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true)
    endif
endif

if 'GCS' in static_plugins
    gcs_backend_interface = declare_dependency(link_whole: gcs_backend_lib)
else
    gcs_backend_interface = declare_dependency(link_with: gcs_backend_lib)
endif
