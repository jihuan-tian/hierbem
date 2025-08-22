SET(TEST_TARGET raw-lapack-dtrsv)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
