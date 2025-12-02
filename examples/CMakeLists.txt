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

# Testing models directory
set(HBEM_TEST_MODEL_DIR ${CMAKE_SOURCE_DIR}/tests/models/)

add_subdirectory(build-cluster-tree)
add_subdirectory(build-block-cluster-tree)
add_subdirectory(build-hmatrix)
add_subdirectory(build-hmatrix-on-subdomain)
add_subdirectory(laplace-dirichlet)
add_subdirectory(laplace-neumann)
add_subdirectory(laplace-mixed)
