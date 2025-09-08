# Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
#
# Copyright (C) 2023-2025 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your option)
# any later version. The full text of the license can be found in the file
# LICENSE at the top level directory of HierBEM.
# ------------------------------------------------------------------------------
#
# Older CMake versions do not support nvcc with c++20 standard.
cmake_minimum_required(VERSION 3.25.2)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
  # Keep compatible with FindCUDA() macros used in dealii
  cmake_policy(SET CMP0146 OLD)
endif()

# Project name
project("hierbem" LANGUAGES CUDA CXX)

# External dependencies
find_package(
  deal.II
  9.4.0
  REQUIRED
  HINTS
  ${deal.II_DIR}
  ${DEAL_II_DIR}
  ../
  ../../
  $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
  message(
    FATAL_ERROR
      "\n"
      "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
      "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
      "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif()

find_package(fmt CONFIG REQUIRED)
find_package(cpptrace CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(tomlplusplus REQUIRED IMPORTED_TARGET tomlplusplus)

find_package(reflectcpp REQUIRED HINTS ${REFLECT_CPP_DIR} $ENV{REFLECT_CPP_DIR})
if(NOT ${reflectcpp_FOUND})
  message(
    FATAL_ERROR
      "\n"
      "*** Could not locate a (sufficiently recent) version of reflect-cpp. ***\n\n"
      "You may want to either pass a flag -DREFLECT_CPP_DIR=/path/to/reflect-cpp to cmake\n"
      "or set an environment variable \"REFLECT_CPP_DIR\" that contains this path."
  )
endif()

#
# Check if we are using mixed build mode
#
set(_mixed_build_mode FALSE)
if(NOT "${DEAL_II_BUILD_TYPE}" MATCHES "${CMAKE_BUILD_TYPE}")
  set(_mixed_build_mode TRUE)
endif()

#
# Enforce application using compiling/linking flags of deal.ii library only if
# not in mixed build mode. Useful for linking Debug mode application with
# Release mode deal.ii library to speed up gdb loading.
#
if(NOT _mixed_build_mode)
  deal_ii_initialize_cached_variables()
endif()

# Get CUDA toolkit versions
find_package(CUDAToolkit)

# Include additional CMake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/macros
     ${CMAKE_SOURCE_DIR}/cmake/modules)

include(macro_add_flags)
include(macro_replace_flag)

# Enable ccache to speed up compilation if possible
include(AutoCCache)

message("=== Default compiler and linker flags configured by deal.ii ===")
message("DEAL_II_CXX_FLAGS: ${DEAL_II_CXX_FLAGS}")
message("DEAL_II_CXX_FLAGS_DEBUG: ${DEAL_II_CXX_FLAGS_DEBUG}")
message("DEAL_II_CXX_FLAGS_RELEASE: ${DEAL_II_CXX_FLAGS_RELEASE}")

message("DEAL_II_CUDA_FLAGS: ${DEAL_II_CUDA_FLAGS}")
message("DEAL_II_CUDA_FLAGS_DEBUG: ${DEAL_II_CUDA_FLAGS_DEBUG}")
message("DEAL_II_CUDA_FLAGS_RELEASE: ${DEAL_II_CUDA_FLAGS_RELEASE}")

message("DEAL_II_LINKER_FLAGS: ${DEAL_II_LINKER_FLAGS}")
message("DEAL_II_LINKER_FLAGS_DEBUG: ${DEAL_II_LINKER_FLAGS_DEBUG}")
message("DEAL_II_LINKER_FLAGS_RELEASE: ${DEAL_II_LINKER_FLAGS_RELEASE}")
message("====================================================")

# Set CXX flags in debug mode oriented for performance profiling in Valgrind. If
# the enabler is not defined, set it to false as the default.
if(NOT DEFINED HBEM_ENABLE_VALGRIND_PROFILE)
  set(HBEM_ENABLE_VALGRIND_PROFILE 0)
endif()

if(HBEM_ENABLE_VALGRIND_PROFILE AND ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("*** Enable Valgrind profile in Debug mode")

  # Disable function inlining for performance profiling in Valgrind.
  add_flags(DEAL_II_CXX_FLAGS_DEBUG "-fno-inline")
  # Replace -O0 with -O1 for performance profiling in Valgrind.
  replace_flag(DEAL_II_CXX_FLAGS_DEBUG "-O0" "-O1")
endif()

# Set CUDA compiler flags.
set(DEAL_II_CUDA_FLAGS_DEBUG "-g -G --default-stream per-thread")
set(DEAL_II_CUDA_FLAGS_RELEASE "--default-stream per-thread")

# Remove the language standard in DEAL_II_* related variables and hand over the
# control to the macro HBEM_STD_COMPILE_FLAGS.
string(REGEX REPLACE "(-+std +|-std=)c\\+\\+[0-9][0-9]" "" DEAL_II_CXX_FLAGS
                     ${DEAL_II_CXX_FLAGS})
string(REGEX REPLACE "(-+std +|-std=)c\\+\\+[0-9][0-9]" "" DEAL_II_CUDA_FLAGS
                     ${DEAL_II_CUDA_FLAGS})

# Remove unknown flag from Clang compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  replace_flag(DEAL_II_CXX_FLAGS_RELEASE "-funroll-all-loops" "")
  replace_flag(DEAL_II_CXX_FLAGS "-Wno-placement-new" "")
  replace_flag(DEAL_II_CXX_FLAGS "-Wno-literal-suffix" "")
  replace_flag(DEAL_II_CXX_FLAGS "-Wno-psabi" "")
  replace_flag(DEAL_II_CXX_FLAGS "-Wno-class-memaccess" "")
endif()

# ##############################################################################
# Configuration for performance profiling in nsight-sys
# ##############################################################################
if(NOT DEFINED HBEM_ENABLE_NSIGHT_PROFILE)
  set(HBEM_ENABLE_NSIGHT_PROFILE 0)
endif()

# N.B. When CUDA is involved, the computational task is usually intensive.
# Therefore, we only run performance profiling in the release model.
if(HBEM_ENABLE_NSIGHT_PROFILE AND ${CMAKE_BUILD_TYPE} STREQUAL "Release")
  message("*** Enable nsight-sys profile in Release mode")

  # Add CXX compiler flag for OS runtime libraries trace in nsight-sys.
  add_flags(DEAL_II_CXX_FLAGS_RELEASE "-funwind-tables")
  # Add CUDA compiler flag for generating line number information for
  # performance profiling.
  add_flags(DEAL_II_CUDA_FLAGS_RELEASE "-lineinfo")
endif()

message("=== Compiler and linker flags after customization ===")
message("DEAL_II_CXX_FLAGS: ${DEAL_II_CXX_FLAGS}")
message("DEAL_II_CXX_FLAGS_DEBUG: ${DEAL_II_CXX_FLAGS_DEBUG}")
message("DEAL_II_CXX_FLAGS_RELEASE: ${DEAL_II_CXX_FLAGS_RELEASE}")

message("DEAL_II_CUDA_FLAGS: ${DEAL_II_CUDA_FLAGS}")
message("DEAL_II_CUDA_FLAGS_DEBUG: ${DEAL_II_CUDA_FLAGS_DEBUG}")
message("DEAL_II_CUDA_FLAGS_RELEASE: ${DEAL_II_CUDA_FLAGS_RELEASE}")

message("DEAL_II_LINKER_FLAGS: ${DEAL_II_LINKER_FLAGS}")
message("DEAL_II_LINKER_FLAGS_DEBUG: ${DEAL_II_LINKER_FLAGS_DEBUG}")
message("DEAL_II_LINKER_FLAGS_RELEASE: ${DEAL_II_LINKER_FLAGS_RELEASE}")
message("====================================================")

# Setup targets for fast switching between debug mode and release mode.
add_custom_target(
  debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMAND ${CMAKE_COMMAND} -E echo "*** Switched to Debug mode."
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMENT "Switching CMAKE_BUILD_TYPE to Debug"
  VERBATIM)

add_custom_target(
  release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMAND ${CMAKE_COMMAND} -E echo "*** Switched to Release mode."
  COMMAND ${CMAKE_COMMAND} -E echo "***"
  COMMENT "Switching CMAKE_BUILD_TYPE to Release"
  VERBATIM)

#
# Set default config values
#
if(NOT DEFINED HBEM_ENABLE_DEBUG)
  set(HBEM_ENABLE_DEBUG 1)
endif()

if(NOT DEFINED HBEM_MESSAGE_LEVEL)
  set(HBEM_MESSAGE_LEVEL 1)
endif()

if(NOT DEFINED HBEM_ENABLE_TIMER)
  set(HBEM_ENABLE_TIMER 1)
endif()

if(NOT DEFINED HBEM_ENABLE_MATRIX_EXPORT)
  set(HBEM_ENABLE_MATRIX_EXPORT 0)
endif()

# Enable print out component matrices involved during building an operator
# preconditioner.
if(NOT DEFINED HBEM_ENABLE_PRECONDITIONER_MATRIX_EXPORT)
  set(HBEM_ENABLE_PRECONDITIONER_MATRIX_EXPORT 0)
endif()

if(NOT DEFINED HBEM_ENABLE_NVTX)
  set(HBEM_ENABLE_NVTX 0)
endif()

# If ACA runs in multiple threads, setting this flag to 0 will not make it
# really deterministic. To fully remove the randomness in ACA, TBB thread number
# should be set to 1 by calling `dealii::MultithreadInfo::set_thread_limit(1)`.
if(NOT DEFINED HBEM_RANDOM_ACA)
  set(HBEM_RANDOM_ACA 0)
endif()

# Do not use TBB task arena or isolate in H-LU or H-Cholesky by default.
if(NOT DEFINED HBEM_ARENA_OR_ISOLATE_IN_LU_AND_CHOL)
  set(HBEM_ARENA_OR_ISOLATE_IN_LU_AND_CHOL 3)
endif()

if(NOT DEFINED HBEM_ARENA_MAX_CONCURRENCY_IN_LU_AND_CHOL)
  set(HBEM_ARENA_MAX_CONCURRENCY_IN_LU_AND_CHOL 1)
endif()

# Select the solution space for Laplace problem with Neumann boundary condition.
# 1: \f$H_{\ast}^{1/2}(\Gamma)\f$; 2: \f$H_{\ast \ast}^{1/2}(\Gamma)\f$
if(NOT DEFINED HBEM_NEUMANN_SOLUTION_SPACE)
  set(HBEM_NEUMANN_SOLUTION_SPACE 2)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/extern/cuda_complex)

# Directories for searching libraries to link
link_directories(${CMAKE_BINARY_DIR}/src /usr/lib/x86_64-linux-gnu)

# Folder for header files and source code
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(scripts)

# Enable unit testing for this deal.ii user project.
enable_testing()

#
# DEAL_II_ADD_TEST() macro enforces application build mode to be the same as the
# linked deal.ii library. Workaround this constraint by temporarily overwrite
# DEAL_II_BUILD_TYPES and restore its value later if mixed build mode is used.
#
if(_mixed_build_mode)
  set(_real_deal_ii_build_types ${DEAL_II_BUILD_TYPES})
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DEAL_II_BUILD_TYPES DEBUG)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(DEAL_II_BUILD_TYPES RELEASE)
  endif()
endif()

# Add unit tests.
add_subdirectory(tests)

#
# ...restore original value of DEAL_II_BUILD_TYPES
#
if(_mixed_build_mode)
  set(DEAL_II_BUILD_TYPES _real_deal_ii_build_types)
endif()
