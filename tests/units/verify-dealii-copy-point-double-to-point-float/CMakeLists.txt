# Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
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
set(_target test-verify-dealii-copy-point-double-to-point-float)

add_executable(${_target} verify-dealii-copy-point-double-to-point-float.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

target_link_libraries(${_target} Catch2::Catch2WithMain)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})
