SET(TEST_TARGET hmatrix-hmatrix-mTmult-level-conserving-all-fine-ntp)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
