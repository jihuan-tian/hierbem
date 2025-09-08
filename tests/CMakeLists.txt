# Copyright (C) 2023 Jihuan Tian <jihuan_tian@hotmail.com>
# Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
#
# This file is part of the HierBEM library.
#
# HierBEM is free software: you can use it, redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. The full text of the license can be found in the
# file LICENSE at the top level directory of HierBEM.

include(macro_load_test_deps)
include(macro_sync_test_list)
include(macro_hbem_std_compile_flags)

set(_test_list_file ${CMAKE_CURRENT_LIST_DIR}/TestCases.cmake)

# Update testcase list
sync_test_list(${_test_list_file} baseline/ units/ functions/ regressions/
               demos/)

# Load testing dependencies
load_test_deps()

# Generate testing config header files
set(HBEM_ROOT_DIR ${CMAKE_SOURCE_DIR}/) # HierBEM source root
set(HBEM_TEST_MODEL_DIR ${CMAKE_SOURCE_DIR}/tests/models/) # Testing models
                                                           # directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/hbem_test_config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/hbem_test_config.h)

# Add testing config header files to include path
include_directories(APPEND ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Enable more strict compilation for C++ and CUDA
hbem_std_compile_flags()

add_subdirectory(src)

# Add all testcase subdirectories
include(${_test_list_file})

# Add custom target to cleanup deal.ii test output files
add_custom_target(
  clean_output
  COMMAND find -name output -delete
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  VERBATIM)
