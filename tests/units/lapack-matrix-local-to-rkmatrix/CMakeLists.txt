SET(TEST_TARGET lapack-matrix-local-to-rkmatrix)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools)

DEAL_II_PICKUP_TESTS()
