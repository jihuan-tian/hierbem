set(_target vmult-serial)

add_executable(${_target} vmult-serial.cu)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${_target} generic_tools boost_program_options dl)
