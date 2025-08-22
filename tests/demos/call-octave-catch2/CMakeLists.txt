find_package(Octave REQUIRED)

# Target should contains 'test' substring in any cases in order to be found by
# Visual Studio Code's TestMate extension
set(_target test-call-octave-catch2)
add_executable(${_target} call-octave-catch2.cc)

# XXX Relax compiler's pedantic checks to include buggy Octave headers
set_target_properties(${_target} PROPERTIES HBEM_CHECK_RELAXED 1)

# Allow target to include Octave header files
target_include_directories(${_target} PRIVATE ${OCTAVE_INCLUDE_DIRS})

# Define SOURCE_DIR macro to be used in the source files
target_compile_definitions(${_target}
                           PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${_target} hierbem boost_program_options dl
                      ${OCTAVE_LIBRARIES} Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(${_target})
