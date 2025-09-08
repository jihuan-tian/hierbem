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
 * \file stl-gen-linear-indices.cc
 * \brief
 * \ingroup test_cases stl
 * \author Jihuan Tian
 * \date 2022-03-10
 */

#include "utilities/debug_tools.h"
#include "utilities/generic_functors.h"
#include "utilities/unary_template_arg_containers.h"

using namespace std;

int
main()
{
  vector<int> a(10);
  //! The two template argument should be given explicitly.
  gen_linear_indices<vector_uta, int>(a, 0, 2);
  print_vector_values(std::cout, a, " ", false);
  cout << endl;

  forward_list<unsigned int> b(10);
  gen_linear_indices<forward_list_uta, unsigned int>(b, 1, 1);
  print_vector_values(std::cout, b, " ", false);
  cout << endl;

  list<unsigned int> c(10);
  gen_linear_indices<list_uta, unsigned int>(c, 1, 2);
  print_vector_values(std::cout, c, " ", false);
  cout << endl;

  return 0;
}
