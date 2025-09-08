# Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

set(TEST_TARGET create-efield-subdomains)

# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES
    hierbem
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

deal_ii_pickup_tests()
