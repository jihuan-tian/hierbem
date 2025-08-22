set(TEST_TARGET aca-full-matrix-approximation)

# Link to external libraries.
link_directories("/usr/lib/x86_64-linux-gnu")
# Link TEST_TARGET to hierbem library.
set(TEST_LIBRARIES boost_program_options hierbem)

deal_ii_pickup_tests()
