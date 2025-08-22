SET(TEST_TARGET rkmatrix-agglomeration-rank0)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
