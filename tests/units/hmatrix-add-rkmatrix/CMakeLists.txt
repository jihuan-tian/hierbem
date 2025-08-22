SET(TEST_TARGET hmatrix-add-rkmatrix)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
