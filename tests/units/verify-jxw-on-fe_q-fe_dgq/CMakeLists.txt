SET(TEST_TARGET verify-jxw-on-fe_q-fe_dgq)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
