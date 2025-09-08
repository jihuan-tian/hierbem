# Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-reflectcpp-usage)

add_executable(${_target} reflectcpp-usage.cc)

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
  ${_target}
  boost_program_options
  dl # Lazy way: always link with libdl no matter if CUDA is used or not
  reflectcpp::reflectcpp
  Catch2::Catch2WithMain
  PkgConfig::tomlplusplus)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})
