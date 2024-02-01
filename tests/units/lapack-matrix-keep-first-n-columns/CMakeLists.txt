set(TEST_TARGET lapack-matrix-keep-first-n-columns)

# Link TEST_TARGET to my generic_tools library.
set(TEST_LIBRARIES
    generic_tools
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

deal_ii_pickup_tests()
