include(macro_hbem_std_compile_flags)

# ##############################################################################
# Library target: hierbem
# ##############################################################################

# Enable more strict compilation for C++ and CUDA
hbem_std_compile_flags()

set(_target hierbem)

add_library(
  ${_target} SHARED
  debug_tools.cc
  dof_tools_ext.cc
  hmatrix/aca_plus/aca_plus.cu
  hmatrix/hmatrix_support.cc
  generic_functors.cc
  sauter_quadrature.cu
  regex_tools.cc
  laplace_bem.cu)
# Because hierbem has included deal.ii header files, dependent include paths and
# library paths need to be imported by calling this macro DEAL_II_SETUP_TARGET.
deal_ii_setup_target(${_target})

# HMatrix class depends on openblas.
target_link_libraries(${_target} openblas)
