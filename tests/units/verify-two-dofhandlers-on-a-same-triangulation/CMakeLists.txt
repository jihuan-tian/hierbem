SET(TEST_TARGET verify-two-dofhandlers-on-a-same-triangulation)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
