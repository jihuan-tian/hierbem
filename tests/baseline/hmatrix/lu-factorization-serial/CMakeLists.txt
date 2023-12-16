find_package(Octave REQUIRED)

set(_target lu-factorization-serial)

add_executable(${_target} lu-factorization-serial.cu)

# Allow target to include deal.ii header files
deal_ii_setup_target(${_target})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${_target} hbem_octave_wrapper generic_tools
                      boost_program_options dl ${OCTAVE_LIBRARIES})
