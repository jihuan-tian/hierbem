SET(TEST_TARGET verify-integration-on-cylinder-hull)

# Link TEST_TARGET to hierbem library as well as dl library for NVTX
# instrumentation.
SET(TEST_LIBRARIES hierbem boost_program_options dl)

DEAL_II_PICKUP_TESTS()
