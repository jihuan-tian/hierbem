# This module defines:
# JULIA_EXECUTABLE  - Julia executable
# JULIA_INCLUDE_DIR - include path for julia.h
# JULIA_LIBRARY     - Julia library libjulia
# JULIA_LIBRARY_DIR - Path to the Julia library libjulia

# Find Julia executable
find_program(
  JULIA_EXECUTABLE
  NAMES julia
  HINTS ${JULIA_DIR}/bin
  DOC "Julia executable")

if(JULIA_EXECUTABLE)
  execute_process(
    COMMAND ${JULIA_EXECUTABLE} --version
    OUTPUT_VARIABLE JULIA_VERSION_STRING
    RESULT_VARIABLE RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(RESULT EQUAL 0)
    string(REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
                         JULIA_VERSION_STRING ${JULIA_VERSION_STRING})
    set(JULIA_VERSION
        ${JULIA_VERSION_STRING}
        CACHE STRING "Julia version")
  endif()

  execute_process(
    COMMAND
      ${JULIA_EXECUTABLE} -E
      "joinpath(match(r\"(.*)bin\", Sys.BINDIR).captures[1], \"include\", \"julia\")"
    OUTPUT_VARIABLE JULIA_INCLUDE_PATH
    RESULT_VARIABLE RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(RESULT EQUAL 0)
    string(REGEX REPLACE "\"" "" JULIA_INCLUDE_PATH ${JULIA_INCLUDE_PATH})
    set(JULIA_INCLUDE_DIR
        ${JULIA_INCLUDE_PATH}
        CACHE PATH "Julia include directory")
  endif()

  execute_process(
    COMMAND
      ${JULIA_EXECUTABLE} -E
      "using Libdl; abspath(dirname(Libdl.dlpath(Libdl.dlopen(\"libjulia\"))))"
    OUTPUT_VARIABLE JULIA_LIBRARY_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(RESULT EQUAL 0)
    string(REGEX REPLACE "\"" "" JULIA_LIBRARY_PATH ${JULIA_LIBRARY_PATH})
    set(JULIA_LIBRARY_DIR
        ${JULIA_LIBRARY_PATH}
        CACHE PATH "Julia library directory")
  endif()

  find_library(
    JULIA_LIBRARY
    NAMES julia libjulia
    PATHS ${JULIA_LIBRARY_DIR})
endif()

# handle REQUIRED and QUIET options
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Julia
  REQUIRED_VARS JULIA_LIBRARY JULIA_LIBRARY_DIR JULIA_INCLUDE_DIR
  VERSION_VAR JULIA_VERSION
  FAIL_MESSAGE "Julia not found")

mark_as_advanced(JULIA_LIBRARY JULIA_LIBRARY_DIR JULIA_INCLUDE_DIR
                 JULIA_VERSION)
