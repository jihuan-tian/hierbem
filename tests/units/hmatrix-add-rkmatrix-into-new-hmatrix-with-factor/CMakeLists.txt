SET(TEST_TARGET hmatrix-add-rkmatrix-into-new-hmatrix-with-factor)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools)

DEAL_II_PICKUP_TESTS()
