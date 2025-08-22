SET(TEST_TARGET hmatrix-block-triangular-forward-substitution-unit-diag)

# Link TEST_TARGET to hierbem library.
SET(TEST_LIBRARIES hierbem)

DEAL_II_PICKUP_TESTS()
