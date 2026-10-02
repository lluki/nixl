# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

if absl_synchronization_dep.type_name() == 'internal'
  error('NIXLShard needs shared Abseil dependencies: the static wrap duplicates mutex state across NIXL DSOs. Use tools/build-dev.sh or configure a shared Abseil prefix with PKG_CONFIG_PATH.')
endif

nixlshard_inc = include_directories('include')
nixlshard_lib = shared_library('nixlshard',
  ['src/storage.cpp', 'src/agent.cpp', 'src/wire.cpp', 'src/metadata.cpp'],
  include_directories: nixlshard_inc,
  dependencies: [nixl_dep, nixl_infra, thread_dep],
  install: true,
  install_rpath: join_paths(get_option('prefix'), get_option('libdir')))
nixlshard_dep = declare_dependency(
  link_with: nixlshard_lib,
  include_directories: nixlshard_inc,
  dependencies: thread_dep)

install_headers(
  'include/nixlshard/types.h', 'include/nixlshard/storage.h',
  'include/nixlshard/agent.h', 'include/nixlshard/wire.h',
  subdir: 'nixlshard')
import('pkgconfig').generate(nixlshard_lib,
  name: 'NIXLShard', description: 'NIXL distributed cache extension',
  version: '0.1.0', filebase: 'nixlshard')

nixlshard_git = run_command('git', 'rev-parse', '--short=12', 'HEAD', check: true).stdout().strip()
py.extension_module('_bindings', 'bindings/python/bindings.cpp',
  subdir: 'nixlshard', dependencies: [nixlshard_dep, pybind_dep],
  cpp_args: ['-DNIXLSHARD_BUILD_GIT="' + nixlshard_git + '"'],
  install: true,
  install_rpath: join_paths(get_option('prefix'), get_option('libdir')))
py.install_sources('bindings/python/nixlshard/__init__.py', subdir: 'nixlshard')

foreach tool : ['agent', 'metadata']
  executable('nixlshard-' + tool, 'tools/nixlshard-' + tool + '.cpp',
    dependencies: nixlshard_dep, install: true,
    install_rpath: join_paths(get_option('prefix'), get_option('libdir')))
endforeach

if get_option('build_tests')
  nixlshard_test_env = environment()
  nixlshard_test_env.set('NIXL_PLUGIN_DIR',
    join_paths(get_option('prefix'), get_option('libdir'), 'plugins'))
  nixlshard_test_env.set('UCX_TLS', 'tcp,self,cuda_copy')
  foreach name : ['storage', 'agent', 'metadata', 'remote', 'timeout', 'admission']
    exe = executable('nixlshard-' + name + '-test', 'tests/' + name + '_test.cpp',
      dependencies: nixlshard_dep)
    test('nixlshard-' + name, exe, suite: 'nixlshard',
      env: nixlshard_test_env, timeout: 120)
  endforeach
  wire_test = executable('nixlshard-wire-test',
    ['tests/wire_test.cpp', 'src/wire.cpp'],
    include_directories: nixlshard_inc, dependencies: thread_dep)
  test('nixlshard-wire', wire_test, suite: 'nixlshard', timeout: 120)
  cancellation_test = executable('nixlshard-cancellation-test', 'tests/cancellation_test.cpp',
    dependencies: [nixlshard_dep, dl_dep], link_args: ['-Wl,--export-dynamic'])
  test('nixlshard-cancellation', cancellation_test, suite: 'nixlshard',
    env: nixlshard_test_env, timeout: 120)
endif
