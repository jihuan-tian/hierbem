# Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
# Copyright (C) 2023-2024 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version. The full text of the license can be found in the file
# LICENSE at the top level directory of HierBEM.

include(macro_hbem_std_compile_flags)

# ##############################################################################
# Library target: hierbem
# ##############################################################################

# Enable more strict compilation for C++ and CUDA
hbem_std_compile_flags()

set(_target hierbem)

add_library(
  ${_target} SHARED
  utilities/debug_tools.cc
  dofs/dof_tools_ext.cc
  hmatrix/aca_plus/aca_plus.cu
  hmatrix/hmatrix_support.cc
  quadrature/sauter_quadrature.cu
  utilities/regex_tools.cc
  laplace/laplace_bem.cu)
# Because hierbem has included deal.ii header files, dependent include paths and
# library paths need to be imported by calling this macro DEAL_II_SETUP_TARGET.
deal_ii_setup_target(${_target})

# HMatrix class depends on openblas.
target_link_libraries(${_target} openblas)
