SET(TEST_TARGET verify-deterministic-random-numbers)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES
    generic_tools
    boost_program_options
    dl  # Lazy way: always link with libdl no matter if CUDA is used or not
)

DEAL_II_PICKUP_TESTS()
