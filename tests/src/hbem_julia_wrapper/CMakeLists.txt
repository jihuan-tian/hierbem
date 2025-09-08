# Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

set(_target hbem_julia_wrapper)

add_library(${_target} SHARED ${_target}.cc)
target_include_directories(${_target} PUBLIC ${JULIA_DIR}/include/julia)
target_link_directories(${_target} PUBLIC ${JULIA_DIR}/lib)
target_link_libraries(${_target} PUBLIC julia)
