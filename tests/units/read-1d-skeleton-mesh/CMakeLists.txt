set(TEST_TARGET read-1d-skeleton-mesh)

# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES
    hierbem
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

deal_ii_pickup_tests()
