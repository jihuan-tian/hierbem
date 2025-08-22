find_package(Octave REQUIRED)

# Allow target to include Octave header files
include_directories(${OCTAVE_INCLUDE_DIRS})

set(TEST_TARGET call-octave-dealii)

# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES hierbem boost_program_options dl ${OCTAVE_LIBRARIES})

# XXX Relax compiler's pedantic checks to include buggy Octave headers. deal.ii
# testing framework could generate multiple actual targets based on the given
# TEST_TARGET. So we can only relax globally
set(HBEM_CHECK_RELAXED 1)

deal_ii_pickup_tests()
