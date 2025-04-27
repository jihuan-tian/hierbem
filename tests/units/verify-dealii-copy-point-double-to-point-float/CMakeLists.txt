# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-verify-dealii-copy-point-double-to-point-float)

add_executable(${_target} verify-dealii-copy-point-double-to-point-float.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

target_link_libraries(${_target} Catch2::Catch2WithMain)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})
