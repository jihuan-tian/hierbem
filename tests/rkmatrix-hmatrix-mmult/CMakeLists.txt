SET(TEST_TARGET rkmatrix-hmatrix-mmult)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools)

DEAL_II_PICKUP_TESTS()
