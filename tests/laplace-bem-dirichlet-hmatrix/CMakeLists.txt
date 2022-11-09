SET(TEST_TARGET laplace-bem-dirichlet-hmatrix)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools boost_program_options)

DEAL_II_PICKUP_TESTS()

# # Write debug symbols into the release binary.
# include(${DEAL_II_DIR}/cmake/macros/macro_add_flags.cmake)
# # N.B. Use -ggdb instead of -g
# ADD_FLAGS(CMAKE_CXX_FLAGS "-ggdb")
# ADD_FLAGS(CMAKE_CXX_FLAGS "-fno-inline")

# Post-build event
ADD_CUSTOM_COMMAND(TARGET laplace-bem-dirichlet-hmatrix.debug POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
ADD_CUSTOM_COMMAND(TARGET laplace-bem-dirichlet-hmatrix.release POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
