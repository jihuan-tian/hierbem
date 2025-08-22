SET(TEST_TARGET hmatrix-hmatrix-mmult-all-fine-ntp-with-adding)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
