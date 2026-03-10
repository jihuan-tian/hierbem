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

if(NOT DEFINED HBEM_USE_64BIT_INDICES)
  set(HBEM_USE_64BIT_INDICES 0)
endif()

if(HBEM_USE_64BIT_INDICES)
  set(MKL_INTERFACE ilp64)
else()
  set(MKL_INTERFACE lp64)
endif()

find_package(MKL CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

set(_target hierbem)

add_library(
  ${_target} SHARED
  config_file/cu_related.cu
  utilities/debug_tools.cc
  dofs/dof_tools_ext.cc
  hmatrix/aca_plus/aca_plus.cu
  hmatrix/hmatrix_support.cc
  quadrature/sauter_quadrature.cu
  utilities/regex_tools.cc
  laplace/laplace_bem.cu
  helmholtz/helmholtz_acoustic_bem.cu)

# Because my library hierbem has included deal.ii header files, dependent
# include paths and library paths need to be imported by calling this macro
# DEAL_II_SETUP_TARGET.
deal_ii_setup_target(${_target})

set_target_properties(${_target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# HMatrix class depends on MKL.
target_link_libraries(
  ${_target}
  dl # Lazy way: always link with libdl no matter if CUDA is used or not
  MKL::MKL
  TBB::tbb
  fmt::fmt
  reflectcpp::reflectcpp
  PkgConfig::tomlplusplus)

add_subdirectory(hbem_laplace)
