# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-verify-dealii-real-sparsematrix-vmult-complex-vector)

add_executable(${_target}
               verify-dealii-real-sparsematrix-vmult-complex-vector.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

target_link_libraries(${_target} hierbem hbem_julia_wrapper
                      Catch2::Catch2WithMain)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/process.jl
          ${CMAKE_CURRENT_BINARY_DIR}/process.jl)
