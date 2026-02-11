# Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

# Target should contain 'test' substring in any case in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-cuda-complex-std-compat)

add_executable(${_target} cuda-complex-std-compat.cc)

target_include_directories(${_target} PRIVATE ${CMAKE_SOURCE_DIR}/extern/cuda_complex)

target_link_libraries(${_target} Catch2::Catch2WithMain)

# Discover Catch2 test cases
include(CTest)
include(Catch)
catch_discover_tests(${_target})