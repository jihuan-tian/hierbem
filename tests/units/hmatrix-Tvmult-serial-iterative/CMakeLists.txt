# Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

find_package(Octave REQUIRED)

# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-hmatrix-Tvmult-serial-iterative)

add_executable(${_target} hmatrix-Tvmult-serial-iterative.cc
                          run-hmatrix-Tvmult-serial-iterative.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
  ${_target}
  hbem_octave_wrapper
  hierbem
  boost_program_options
  dl
  ${OCTAVE_LIBRARIES}
  Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(${_target})

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/process.m
          ${CMAKE_CURRENT_BINARY_DIR}/process.m)
