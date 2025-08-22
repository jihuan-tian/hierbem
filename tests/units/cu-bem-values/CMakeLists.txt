SET(TEST_TARGET cu-bem-values)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
