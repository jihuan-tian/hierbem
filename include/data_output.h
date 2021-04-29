// File: data_output.h
// Description: Header file for handling data output.
// Author: Jihuan Tian
// Date: 2020-11-10
// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>

#ifndef INCLUDE_DATA_OUTPUT_H_
#define INCLUDE_DATA_OUTPUT_H_

#include <vector>
#include <string>
#include <iostream>

namespace LaplaceBEM
{
  template<typename T>
  void print_vector(std::ostream &out, const std::vector<T> &input_vector, std::string separator)
  {
    for (auto element : input_vector)
    {
      out << element << separator;
    }

    out << std::endl;
  }
}

#endif /* INCLUDE_DATA_OUTPUT_H_ */
