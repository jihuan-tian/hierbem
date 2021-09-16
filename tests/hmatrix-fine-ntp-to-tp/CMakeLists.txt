SET(TEST_TARGET hmatrix-fine-ntp-to-tp)

# Link TEST_TARGET to my generic_tools library.
SET(TEST_LIBRARIES generic_tools)

DEAL_II_PICKUP_TESTS()

ADD_CUSTOM_COMMAND(TARGET hmatrix-fine-ntp-to-tp.debug POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
ADD_CUSTOM_COMMAND(TARGET hmatrix-fine-ntp-to-tp.release POST_BUILD
  COMMAND mplayer ${CMAKE_SOURCE_DIR}/media/Oxygen-Im-Nudge.ogg > /dev/null 2>&1
  VERBATIM)
