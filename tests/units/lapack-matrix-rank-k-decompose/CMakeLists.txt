# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-lapack-matrix-rank-k-decompose)

add_executable(${_target} lapack-matrix-rank-k-decompose.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

target_link_libraries(
  ${_target}
  hierbem
  boost_program_options
  dl # Lazy way: always link with libdl no matter if CUDA is used or not
  Catch2::Catch2WithMain)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})
