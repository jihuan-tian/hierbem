SET(TEST_TARGET verify-get-vertex-indices-via-face-iter)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools boost_program_options)

DEAL_II_PICKUP_TESTS()

# Post-build event
ADD_CUSTOM_COMMAND(TARGET verify-get-vertex-indices-via-face-iter.debug POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
ADD_CUSTOM_COMMAND(TARGET verify-get-vertex-indices-via-face-iter.release POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
