# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

include(macro_hbem_std_compile_flags)

# Enable more strict compilation for C++ and CUDA
hbem_std_compile_flags()

# Add testing config header files to include path
include_directories(APPEND ${CMAKE_SOURCE_DIR}/tests/include)

set(_target build-hmatrix-on-subdomain)
add_executable(${_target} build-hmatrix-on-subdomain.cu)
deal_ii_setup_target(${_target})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plot-hmat.asy.in
               ${CMAKE_CURRENT_SOURCE_DIR}/plot-hmat.asy)

target_link_libraries(${_target} hierbem)

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/plot-hmat.asy
          ${CMAKE_CURRENT_BINARY_DIR}/plot-hmat.asy)
