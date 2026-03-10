# Copyright (C) 2025 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

include(macro_hbem_std_compile_flags)
hbem_std_compile_flags()

set(_target hbem_laplace)
add_executable(${_target} hbem_laplace.cc)
target_link_libraries(${_target} hierbem fmt::fmt reflectcpp::reflectcpp
                      PkgConfig::tomlplusplus cpptrace::cpptrace)
deal_ii_setup_target(${_target})

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.default.toml
    ${CMAKE_CURRENT_BINARY_DIR}/config.default.toml)
