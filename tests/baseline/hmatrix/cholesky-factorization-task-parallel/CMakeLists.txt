set(_target cholesky-factorization-task-parallel)

add_executable(${_target} cholesky-factorization-task-parallel.cu)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${_target} generic_tools boost_program_options dl
                      ${OCTAVE_LIBRARIES})
