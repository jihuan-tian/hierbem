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

find_package(Octave REQUIRED)

set(_target hbem_octave_wrapper)

# Octave wrapper solib
add_library(${_target} SHARED ${_target}.cc)

# XXX Relax compiler's pedantic checks to include buggy Octave headers
set_target_properties(${_target} PROPERTIES HBEM_CHECK_RELAXED 1)

target_include_directories(${_target} PUBLIC ${OCTAVE_INCLUDE_DIRS})
target_link_libraries(${_target} PUBLIC ${OCTAVE_LIBRARIES})
