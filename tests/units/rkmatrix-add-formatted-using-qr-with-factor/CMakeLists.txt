SET(TEST_TARGET rkmatrix-add-formatted-using-qr-with-factor)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools)

DEAL_II_PICKUP_TESTS()
