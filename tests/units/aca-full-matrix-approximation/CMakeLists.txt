# Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

set(TEST_TARGET aca-full-matrix-approximation)

# Link to external libraries.
link_directories("/usr/lib/x86_64-linux-gnu")
# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES boost_program_options hierbem)

deal_ii_pickup_tests()
