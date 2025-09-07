// File: data_output.h
// Description: Header file for handling data output.
// Author: Jihuan Tian
// Date: 2020-11-10
// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>

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
