SET(TEST_TARGET cudafullmatrix-invert-by-gauss-elim)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem boost_program_options)

DEAL_II_PICKUP_TESTS()
