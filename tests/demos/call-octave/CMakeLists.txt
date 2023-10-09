FIND_PACKAGE(Octave REQUIRED)

# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target test-call-octave)

ADD_EXECUTABLE(${_target} call-octave.cc)

# Allow target to include Octave header files
INCLUDE_DIRECTORIES(${OCTAVE_INCLUDE_DIRS})
# Allow target to include deal.ii header files
DEAL_II_SETUP_TARGET(${_target})

# Define SOURCE_DIR macro to be used in the source files
TARGET_COMPILE_DEFINITIONS(${_target} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

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
