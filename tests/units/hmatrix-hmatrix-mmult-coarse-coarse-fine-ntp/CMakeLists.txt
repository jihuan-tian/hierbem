SET(TEST_TARGET hmatrix-hmatrix-mmult-coarse-coarse-fine-ntp)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
