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
 * \file dealii-vector-argmax-argmin.cc
 * \brief
 * \ingroup test_cases
 *
 * \author Jihuan Tian
 * \date 2022-03-10
 */

#include <deal.II/lac/vector.h>

#include <algorithm>
#include <forward_list>
#include <iostream>
#include <iterator>
#include <vector>

using namespace dealii;

int
main()
{
  std::vector<int> a{3, 2, 5, 7, 1, 9, 0};
  Vector<int>      v(a.begin(), a.end());

  auto iter1 = std::max_element(v.begin(), v.end());
  std::cout << "Max element index: " << iter1 - v.begin() << std::endl;
  iter1 = std::min_element(v.begin(), v.end());
  std::cout << "Min element index: " << iter1 - v.begin() << std::endl;

  std::forward_list<int> b{3, 2, 5, 7, 1, 9, 0};
  auto                   iter2 = std::max_element(b.begin(), b.end());
  std::cout << "Max element index in forward list: "
            << std::distance(b.begin(), iter2) << std::endl;
  iter2 = std::min_element(b.begin(), b.end());
  std::cout << "Min element index in forward list: "
            << std::distance(b.begin(), iter2) << std::endl;

  return 0;
}
