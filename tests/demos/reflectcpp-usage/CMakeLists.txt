# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-reflectcpp-usage)

add_executable(${_target} reflectcpp-usage.cc)

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(tomlplusplus REQUIRED IMPORTED_TARGET tomlplusplus)

target_link_libraries(
  ${_target}
  generic_tools
  boost_program_options
  dl # Lazy way: always link with libdl no matter if CUDA is used or not
  Catch2::Catch2WithMain
  PkgConfig::tomlplusplus)

# Discover Catch2 testcases
include(CTest)
include(Catch)
catch_discover_tests(${_target})
