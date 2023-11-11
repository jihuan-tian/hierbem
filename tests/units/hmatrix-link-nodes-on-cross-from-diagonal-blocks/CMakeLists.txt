set(TEST_TARGET hmatrix-link-nodes-on-cross-from-diagonal-blocks)

# Link TEST_TARGET to my generic_tools library.
set(TEST_LIBRARIES
    hbem_octave_wrapper
    generic_tools
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

deal_ii_pickup_tests()
