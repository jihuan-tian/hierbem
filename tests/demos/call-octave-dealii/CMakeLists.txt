FIND_PACKAGE(Octave REQUIRED)

# Allow target to include Octave header files
INCLUDE_DIRECTORIES(${OCTAVE_INCLUDE_DIRS})

SET(TEST_TARGET call-octave-dealii)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES
    generic_tools
    boost_program_options
    dl
    ${OCTAVE_LIBRARIES}
)

DEAL_II_PICKUP_TESTS()
