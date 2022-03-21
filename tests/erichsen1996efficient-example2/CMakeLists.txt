SET(TEST_TARGET erichsen1996efficient-example2)

# Link to external libraries.
LINK_DIRECTORIES("/usr/lib/x86_64-linux-gnu")
SET(TEST_LIBRARIES boost_program_options generic_tools)

DEAL_II_PICKUP_TESTS()