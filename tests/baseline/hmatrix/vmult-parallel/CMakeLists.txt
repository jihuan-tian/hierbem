set(_target vmult-parallel)

add_executable(${_target} vmult-parallel.cu)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${_target} openblas generic_tools boost_program_options
                      dl)
