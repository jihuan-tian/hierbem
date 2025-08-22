set(TEST_TARGET lapack-matrix-keep-first-n-columns)

# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES
    hierbem
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

deal_ii_pickup_tests()
