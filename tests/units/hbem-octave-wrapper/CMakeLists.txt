# Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target test-hbem-octave-wrapper)

ADD_EXECUTABLE(${_target} hbem-octave-wrapper.cc)

# Define SOURCE_DIR macro to be used in the source files
TARGET_COMPILE_DEFINITIONS(${_target} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

TARGET_LINK_LIBRARIES(
    ${_target}
    hbem_octave_wrapper
    hierbem
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
    Catch2::Catch2WithMain
)

# Discover Catch2 testcases
INCLUDE(CTest)
INCLUDE(Catch)
CATCH_DISCOVER_TESTS(${_target})
