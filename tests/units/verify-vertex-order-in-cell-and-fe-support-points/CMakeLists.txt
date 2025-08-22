SET(TEST_TARGET verify-vertex-order-in-cell-and-fe-support-points)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
