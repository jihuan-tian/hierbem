SET(TEST_TARGET verify-integration-on-sphere-manifold)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools boost_program_options dl)

DEAL_II_PICKUP_TESTS()
