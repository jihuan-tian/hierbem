# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target test-dirichlet-full-matrix-single-thread)

ADD_EXECUTABLE(${_target} dirichlet-full-matrix-single-thread.cu)

# Allow target to include deal.ii header files
DEAL_II_SETUP_TARGET(${_target})

# Define SOURCE_DIR macro to be used in the source files
TARGET_COMPILE_DEFINITIONS(${_target} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

TARGET_LINK_LIBRARIES(
    ${_target}
    generic_tools
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
    Catch2::Catch2WithMain
)

# Discover Catch2 testcases
INCLUDE(CTest)
INCLUDE(Catch)
CATCH_DISCOVER_TESTS(${_target})
