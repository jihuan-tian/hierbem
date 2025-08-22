SET(TEST_TARGET rkmatrix-local-to-rkmatrix)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
