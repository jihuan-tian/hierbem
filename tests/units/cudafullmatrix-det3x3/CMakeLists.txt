# Target should contains 'test' substring in any cases in order to be
# found by Visual Studio Code's TestMate extension
SET(_target cudafullmatrix-det3x3-test)
ADD_EXECUTABLE(${_target} cudafullmatrix-det3x3.cu)
# Allow target to include deal.ii header files
DEAL_II_SETUP_TARGET(${_target})
TARGET_LINK_LIBRARIES(
    ${_target}
    generic_tools
    boost_program_options
    Catch2::Catch2WithMain
)

INCLUDE(CTest)
INCLUDE(Catch)
catch_discover_tests(${_target})
