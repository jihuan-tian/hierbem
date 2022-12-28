SET(TEST_TARGET hmatrix-hmatrix-mTmult-level-conserving-symm-result)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools boost_program_options)

DEAL_II_PICKUP_TESTS()
