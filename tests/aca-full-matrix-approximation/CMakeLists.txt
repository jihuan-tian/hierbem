SET(TEST_TARGET aca-full-matrix-approximation)

# Link to external libraries.
LINK_DIRECTORIES("/usr/lib/x86_64-linux-gnu")
# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES boost_program_options generic_tools)

DEAL_II_PICKUP_TESTS()

# Post-build event
ADD_CUSTOM_COMMAND(TARGET aca-full-matrix-approximation.debug POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
ADD_CUSTOM_COMMAND(TARGET aca-full-matrix-approximation.release POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
