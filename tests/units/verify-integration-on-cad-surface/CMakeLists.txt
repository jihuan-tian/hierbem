SET(TEST_TARGET verify-integration-on-cad-surface)

# Link TEST_TARGET to my generic_tools library as well as dl library for NVTX
# instrumentation.
SET(TEST_LIBRARIES generic_tools boost_program_options dl)

DEAL_II_PICKUP_TESTS()
