SET(TEST_TARGET hmatrix-truncate-to-fixed-rank-preserve-spd)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
