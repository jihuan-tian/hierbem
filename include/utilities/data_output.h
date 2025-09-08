// Copyright (C) 2020-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file data_output.h
 * @brief Header file for handling data output.
 * @author Jihuan Tian
 * @date 2020-11-10
 */

#ifndef HIERBEM_INCLUDE_UTILITIES_DATA_OUTPUT_H_
#define HIERBEM_INCLUDE_UTILITIES_DATA_OUTPUT_H_

#include <iostream>
#include <string>
#include <vector>

#include "config.h"

HBEM_NS_OPEN

template <typename T>
void
print_vector(std::ostream         &out,
             const std::vector<T> &input_vector,
             std::string           separator)
{
  for (auto element : input_vector)
    {
      out << element << separator;
    }

  out << std::endl;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_UTILITIES_DATA_OUTPUT_H_
