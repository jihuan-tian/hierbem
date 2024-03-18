find_package(Octave REQUIRED)

# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-hmatrix-vmult-tril-parallel)

add_executable(${_target} hmatrix-vmult-tril-parallel.cc
                          run-hmatrix-vmult-tril-parallel.cu)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
  ${_target}
  openblas
  hbem_octave_wrapper
  generic_tools
  boost_program_options
  dl
  ${OCTAVE_LIBRARIES}
  Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(${_target})

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/process.m
          ${CMAKE_CURRENT_BINARY_DIR}/process.m)
