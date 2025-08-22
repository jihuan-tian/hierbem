set(TEST_TARGET op-precond-averaging-matrix-for-dirichlet)

# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES
    hierbem
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

deal_ii_pickup_tests()
