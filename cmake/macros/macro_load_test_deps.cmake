# Copyright (C) 2024-2025 Xiaozhe Wang <chaoslawful@gmail.com>
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
# Load external dependencies for testing
#
# Usage: LOAD_TEST_DEPS()
#
macro(LOAD_TEST_DEPS)
  include(FetchContent)

  #
  # Add bundled Catch2 C++ test framework
  #
  FetchContent_Declare(Catch2 SOURCE_DIR
                              "${CMAKE_SOURCE_DIR}/extern/Catch2-3.4.0/")
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/extern/Catch2-3.4.0/extras)

endmacro()
