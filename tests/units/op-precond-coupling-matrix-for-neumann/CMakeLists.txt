# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-op-precond-coupling-matrix-for-neumann)

add_executable(${_target} op-precond-coupling-matrix-for-neumann.cu)
target_include_directories(${_target} PUBLIC ${JULIA_DIR}/include/julia)
target_link_directories(${_target} PUBLIC ${JULIA_DIR}/lib)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/process.jl.in
               ${CMAKE_CURRENT_SOURCE_DIR}/process.jl)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/draw.jl.in
               ${CMAKE_CURRENT_SOURCE_DIR}/draw.jl)

target_link_libraries(
  ${_target}
  hierbem
  boost_program_options
  dl # Lazy way: always link with libdl no matter if CUDA is used or not
  julia
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

add_custom_command(
  TARGET ${_target}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/draw.jl
          ${CMAKE_CURRENT_BINARY_DIR}/draw.jl)
