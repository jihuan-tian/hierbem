SET(TEST_TARGET hmatrix-build-from-aca-with-mass-matrix-smp)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES boost_program_options generic_tools)

DEAL_II_PICKUP_TESTS()
