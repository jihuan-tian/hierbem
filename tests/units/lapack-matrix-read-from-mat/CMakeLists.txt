SET(TEST_TARGET lapack-matrix-read-from-mat)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
