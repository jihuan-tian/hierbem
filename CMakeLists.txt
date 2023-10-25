CMAKE_MINIMUM_REQUIRED(VERSION 3.18.0)
IF(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
    # Keep compatible with FindCUDA() macros used in dealii
    CMAKE_POLICY(SET CMP0146 OLD)
ENDIF()

FIND_PACKAGE(deal.II 9.4.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Check if we are using mixed build mode
#
SET(_mixed_build_mode FALSE)
IF(NOT "${DEAL_II_BUILD_TYPE}" MATCHES "${CMAKE_BUILD_TYPE}")
    SET(_mixed_build_mode TRUE)
ENDIF()

#
# Enforce application using compiling/linking flags of deal.ii library
# only if not in mixed build mode.
# Useful for linking Debug mode application with Release mode deal.ii library
# to speed up gdb loading.
#
IF(NOT _mixed_build_mode)
    DEAL_II_INITIALIZE_CACHED_VARIABLES()
ENDIF()

# Project name
PROJECT("hierbem" LANGUAGES CUDA CXX)

# Get CUDA toolkit versions
FIND_PACKAGE(CUDAToolkit)

# Include additional CMake modules
LIST(APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake/macros
  ${CMAKE_SOURCE_DIR}/cmake/modules
)

INCLUDE(macro_add_flags)
INCLUDE(macro_replace_flag)

MESSAGE("=== Default compiler and linker flags configured by deal.ii ===")
MESSAGE("DEAL_II_CXX_FLAGS: ${DEAL_II_CXX_FLAGS}")
MESSAGE("DEAL_II_CXX_FLAGS_DEBUG: ${DEAL_II_CXX_FLAGS_DEBUG}")
MESSAGE("DEAL_II_CXX_FLAGS_RELEASE: ${DEAL_II_CXX_FLAGS_RELEASE}")

MESSAGE("DEAL_II_CUDA_FLAGS: ${DEAL_II_CUDA_FLAGS}")
MESSAGE("DEAL_II_CUDA_FLAGS_DEBUG: ${DEAL_II_CUDA_FLAGS_DEBUG}")
MESSAGE("DEAL_II_CUDA_FLAGS_RELEASE: ${DEAL_II_CUDA_FLAGS_RELEASE}")

MESSAGE("DEAL_II_LINKER_FLAGS: ${DEAL_II_LINKER_FLAGS}")
MESSAGE("DEAL_II_LINKER_FLAGS_DEBUG: ${DEAL_II_LINKER_FLAGS_DEBUG}")
MESSAGE("DEAL_II_LINKER_FLAGS_RELEASE: ${DEAL_II_LINKER_FLAGS_RELEASE}")
MESSAGE("====================================================")

# Set CXX flags in debug mode oriented for performance profiling in Valgrind.
# If the enabler is not defined, set it to false as the default.
IF (NOT DEFINED HBEM_ENABLE_VALGRIND_PROFILE)
  SET(HBEM_ENABLE_VALGRIND_PROFILE 0)
ENDIF()

IF (HBEM_ENABLE_VALGRIND_PROFILE AND ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  MESSAGE("*** Enable Valgrind profile in Debug mode")

  # Disable function inlining for performance profiling in Valgrind.
  ADD_FLAGS(DEAL_II_CXX_FLAGS_DEBUG "-fno-inline")
  # Replace -O0 with -O1 for performance profiling in Valgrind.
  REPLACE_FLAG(DEAL_II_CXX_FLAGS_DEBUG "-O0" "-O1")
ENDIF()

# Set CUDA compiler flags.
SET(DEAL_II_CUDA_FLAGS_DEBUG "-g -G --default-stream per-thread")
SET(DEAL_II_CUDA_FLAGS_RELEASE "--default-stream per-thread")

# #########################################################
# Configuration for performance profiling in nsight-sys
# #########################################################
IF (NOT DEFINED HBEM_ENABLE_NSIGHT_PROFILE)
  SET(HBEM_ENABLE_NSIGHT_PROFILE 0)
ENDIF()

# N.B. When CUDA is involved, the computational task is usually intensive.
# Therefore, we only run performance profiling in the release model.
IF (HBEM_ENABLE_NSIGHT_PROFILE AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
  MESSAGE("*** Enable nsight-sys profile in Release mode")

  # Add CXX compiler flag for OS runtime libraries trace in nsight-sys.
  ADD_FLAGS(DEAL_II_CXX_FLAGS_RELEASE "-funwind-tables")
  # Add CUDA compiler flag for generating line number information for
  # performance profiling.
  ADD_FLAGS(DEAL_II_CUDA_FLAGS_RELEASE "-lineinfo")
ENDIF()

MESSAGE("=== Compiler and linker flags after customization ===")
MESSAGE("DEAL_II_CXX_FLAGS: ${DEAL_II_CXX_FLAGS}")
MESSAGE("DEAL_II_CXX_FLAGS_DEBUG: ${DEAL_II_CXX_FLAGS_DEBUG}")
MESSAGE("DEAL_II_CXX_FLAGS_RELEASE: ${DEAL_II_CXX_FLAGS_RELEASE}")

MESSAGE("DEAL_II_CUDA_FLAGS: ${DEAL_II_CUDA_FLAGS}")
MESSAGE("DEAL_II_CUDA_FLAGS_DEBUG: ${DEAL_II_CUDA_FLAGS_DEBUG}")
MESSAGE("DEAL_II_CUDA_FLAGS_RELEASE: ${DEAL_II_CUDA_FLAGS_RELEASE}")

MESSAGE("DEAL_II_LINKER_FLAGS: ${DEAL_II_LINKER_FLAGS}")
MESSAGE("DEAL_II_LINKER_FLAGS_DEBUG: ${DEAL_II_LINKER_FLAGS_DEBUG}")
MESSAGE("DEAL_II_LINKER_FLAGS_RELEASE: ${DEAL_II_LINKER_FLAGS_RELEASE}")
MESSAGE("====================================================")

# Setup targets for fast switching between debug mode and release mode.
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMAND ${CMAKE_COMMAND} -E echo "*** Switched to Debug mode."
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMENT "Switching CMAKE_BUILD_TYPE to Debug"
  VERBATIM
)

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMAND ${CMAKE_COMMAND} -E echo "*** Switched to Release mode."
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMENT "Switching CMAKE_BUILD_TYPE to Release"
  VERBATIM
)

# Include directories
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# Directories for searching libraries to link
LINK_DIRECTORIES(${CMAKE_BINARY_DIR}/src /usr/lib/x86_64-linux-gnu)

# Folder for header files and source code
ADD_SUBDIRECTORY(include)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(scripts)

# Enable unit testing for this deal.ii user project.
ENABLE_TESTING()

#
# DEAL_II_ADD_TEST() macro enforces application build mode to be the same as
# the linked deal.ii library.
# Workaround this constraint by temporarily overwrite DEAL_II_BUILD_TYPES
# and restore its value later if mixed build mode is used.
#
IF(_mixed_build_mode)
    SET(_real_deal_ii_build_types ${DEAL_II_BUILD_TYPES})
    IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        SET(DEAL_II_BUILD_TYPES DEBUG)
    ELSEIF("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        SET(DEAL_II_BUILD_TYPES RELEASE)
    ENDIF()
ENDIF()

# Add unit tests.
ADD_SUBDIRECTORY(tests)

#
# ...restore original value of DEAL_II_BUILD_TYPES
#
IF(_mixed_build_mode)
    SET(DEAL_II_BUILD_TYPES _real_deal_ii_build_types)
ENDIF()
