# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-compare-number-types)

add_executable(${_target} compare-number-types.cc)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
  ${_target}
  generic_tools
  boost_program_options
  dl # Lazy way: always link with libdl no matter if CUDA is used or not
  Catch2::Catch2WithMain)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})
