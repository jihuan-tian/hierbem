# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-op-precond-hmatrix-for-dirichlet)

add_executable(${_target} op-precond-hmatrix-for-dirichlet.cc run-op-precond-hmatrix-for-dirichlet.cu)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source filse
target_compile_definitions(${_target} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/process.m.in
               ${CMAKE_CURRENT_SOURCE_DIR}/process.m)

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
