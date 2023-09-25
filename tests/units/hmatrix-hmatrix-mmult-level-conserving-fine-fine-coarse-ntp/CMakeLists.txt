SET(TEST_TARGET hmatrix-hmatrix-mmult-level-conserving-fine-fine-coarse-ntp)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools)

DEAL_II_PICKUP_TESTS()
