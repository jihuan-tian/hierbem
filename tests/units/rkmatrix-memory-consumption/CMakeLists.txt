SET(TEST_TARGET rkmatrix-memory-consumption)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
