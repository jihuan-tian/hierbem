# Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
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
set(_target test-call-octave-catch2)
add_executable(${_target} call-octave-catch2.cc)

# XXX Relax compiler's pedantic checks to include buggy Octave headers
set_target_properties(${_target} PROPERTIES HBEM_CHECK_RELAXED 1)

# Allow target to include Octave header files
target_include_directories(${_target} PRIVATE ${OCTAVE_INCLUDE_DIRS})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${_target} hierbem boost_program_options dl
                      ${OCTAVE_LIBRARIES} Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(${_target})
