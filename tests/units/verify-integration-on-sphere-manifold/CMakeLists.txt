SET(TEST_TARGET verify-integration-on-sphere-manifold)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options dl)

DEAL_II_PICKUP_TESTS()
