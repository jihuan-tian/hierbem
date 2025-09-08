# Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

find_package(fmt CONFIG REQUIRED)

set(_target test-fmt-output)
add_executable(${_target} fmt-output.cc)
target_link_libraries(${_target} PRIVATE fmt::fmt Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(${_target})
