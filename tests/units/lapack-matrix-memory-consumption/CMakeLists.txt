set(TEST_TARGET lapack-matrix-memory-consumption)

# Link TEST_TARGET to my generic_tools library.
set(TEST_LIBRARIES generic_tools)

deal_ii_pickup_tests()
