#
# Load external dependencies for testing
#
# Usage:
#   LOAD_TEST_DEPS()
#
MACRO(LOAD_TEST_DEPS)
    Include(FetchContent)

    #
    # Add bundled Catch2 C++ test framework
    #
    FetchContent_Declare(
        Catch2
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/extern/Catch2-3.4.0/"
    )
    FetchContent_MakeAvailable(Catch2)
    LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2-3.4.0/extras)
ENDMACRO()
