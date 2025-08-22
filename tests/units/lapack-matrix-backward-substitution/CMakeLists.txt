SET(TEST_TARGET lapack-matrix-backward-substitution)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
