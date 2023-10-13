FIND_PACKAGE(Octave REQUIRED)

# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target test-call-octave-catch2)
ADD_EXECUTABLE(${_target} call-octave-catch2.cc)

# XXX Relax compiler's pedantic checks to include buggy Octave headers
SET_TARGET_PROPERTIES(${_target} PROPERTIES HBEM_CHECK_RELAXED 1)

# Allow target to include Octave header files
TARGET_INCLUDE_DIRECTORIES(${_target} PRIVATE ${OCTAVE_INCLUDE_DIRS})

# Define SOURCE_DIR macro to be used in the source files
TARGET_COMPILE_DEFINITIONS(${_target} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

TARGET_LINK_LIBRARIES(
    ${_target}
    generic_tools
    boost_program_options
    dl
    ${OCTAVE_LIBRARIES}
    Catch2::Catch2WithMain
)

INCLUDE(CTest)
INCLUDE(Catch)
CATCH_DISCOVER_TESTS(${_target})
