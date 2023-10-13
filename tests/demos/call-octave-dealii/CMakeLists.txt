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

# XXX Relax compiler's pedantic checks to include buggy Octave headers.
# deal.ii testing framework could generate multiple actual targets
# based on the given TEST_TARGET. So we can only relax globally
SET(HBEM_CHECK_RELAXED 1)

DEAL_II_PICKUP_TESTS()

