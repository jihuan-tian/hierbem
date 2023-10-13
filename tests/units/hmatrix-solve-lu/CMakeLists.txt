FIND_PACKAGE(Octave REQUIRED)

# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target test-hmatrix-solve-lu)

ADD_EXECUTABLE(${_target} hmatrix-solve-lu.cc run-hmatrix-solve-lu.cu)

# Allow target to include deal.ii header files
DEAL_II_SETUP_TARGET(${_target})

# Define SOURCE_DIR macro to be used in the source files
TARGET_COMPILE_DEFINITIONS(${_target} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

TARGET_LINK_LIBRARIES(
    ${_target}
    hbem_octave_wrapper
    generic_tools
    boost_program_options
    dl
    ${OCTAVE_LIBRARIES}
    Catch2::Catch2WithMain
)

INCLUDE(CTest)
INCLUDE(Catch)
CATCH_DISCOVER_TESTS(${_target})

# Copy plotting util to build directory to ease manually post-processing
ADD_CUSTOM_COMMAND(
    TARGET ${_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/draw.m
            ${CMAKE_CURRENT_BINARY_DIR}/draw.m
)
