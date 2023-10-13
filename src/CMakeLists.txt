INCLUDE(macro_hbem_std_compile_flags)

# ##################################
# Library target: generic_tools
# ##################################

# Enable more strict compilation for C++ and CUDA
HBEM_STD_COMPILE_FLAGS()

ADD_LIBRARY(generic_tools debug_tools.cu aca_plus.cu generic_functors.cc sauter_quadrature.cu regex_tools.cc hmatrix_support.cc)
# Because my library generic_tools has included deal.ii header files,
# dependent include paths and library paths need to be imported by
# calling this macro DEAL_II_SETUP_TARGET.
DEAL_II_SETUP_TARGET(generic_tools)
