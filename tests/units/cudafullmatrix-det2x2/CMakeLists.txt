# Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.


# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target cudafullmatrix-det2x2-test)
ADD_EXECUTABLE(${_target} cudafullmatrix-det2x2.cu)

# Allow target to include deal.ii header files
DEAL_II_SETUP_TARGET(${_target})

TARGET_LINK_LIBRARIES(
    ${_target}
    hierbem
    boost_program_options
    Catch2::Catch2WithMain
)

INCLUDE(CTest)
INCLUDE(Catch)
catch_discover_tests(${_target})
