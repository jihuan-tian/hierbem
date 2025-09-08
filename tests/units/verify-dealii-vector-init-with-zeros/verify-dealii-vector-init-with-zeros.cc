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
 * @file verify-dealii-vector-init-with-zeros.cc
 * @brief Verify if a @p dealii::Vector is assigned with zeros after its
 * construction or reinitialization.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2022-12-04
 */

#include <deal.II/lac/vector.h>

#include <iostream>

#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int N = 10;
  Vector<double>     a(N);

  print_vector_values(std::cout, a, ",", false);
  std::cout << "\n";

  for (unsigned int i = 1; i <= N; i++)
    {
      a(i - 1) = i;
    }

  a.reinit(20);
  print_vector_values(std::cout, a, ",", false);
  std::cout << std::endl;
}
