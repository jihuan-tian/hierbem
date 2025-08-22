SET(TEST_TARGET detect-cell-neighboring-type-for-same-dofhandlers)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
