# Copyright (C) 2021-2026 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

set(_target baseline-cluster-tree)

add_executable(${_target} cluster-tree.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

target_link_libraries(${_target} hierbem Boost::program_options)
