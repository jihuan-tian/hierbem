SET(_TARGET laplace-bem-mixed-spanner)

ADD_EXECUTABLE(${_TARGET} laplace-bem-mixed-spanner.cu)

# Allow target to include deal.ii header files
DEAL_II_SETUP_TARGET(${_TARGET})

# Define SOURCE_DIR macro to be used in the source files
TARGET_COMPILE_DEFINITIONS(${_TARGET} PUBLIC SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

TARGET_LINK_LIBRARIES(
    ${_TARGET}
    generic_tools
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)
