// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file regex_tools.h
 * @brief Declaration of tools for regular expressions.
 *
 * @date 2022-03-16
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_UTILITIES_REGEX_TOOLS_H_
#define HIERBEM_INCLUDE_UTILITIES_REGEX_TOOLS_H_

#include <regex>

#include "config.h"

HBEM_NS_OPEN

namespace RegexTools
{
  using namespace std;

  extern const regex reg_for_file_base_and_ext1;
  extern const regex reg_for_file_base_and_ext2;

  string
  file_basename(const string &filename);

  string
  file_ext(const string &filename);
} // namespace RegexTools

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_UTILITIES_REGEX_TOOLS_H_
