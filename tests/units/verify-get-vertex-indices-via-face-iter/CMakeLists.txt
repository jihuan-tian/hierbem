SET(TEST_TARGET verify-get-vertex-indices-via-face-iter)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
