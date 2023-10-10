SET(TEST_TARGET hmatrix-build-from-aca-smp)

# Link to external libraries.
LINK_DIRECTORIES("/usr/lib/x86_64-linux-gnu")
# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES boost_program_options generic_tools)

DEAL_II_PICKUP_TESTS()
