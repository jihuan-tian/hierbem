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
 * @file verify-dealii-vector-memory.cc
 * @brief Verify the internal memory of @p dealii::Vector.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2022-12-04
 */

#include <deal.II/lac/vector.h>

#include <iostream>

using namespace dealii;

int
main()
{
  const unsigned int N = 10;
  Vector<double>     a(N);

  for (unsigned int i = 1; i <= N; i++)
    {
      a(i - 1) = i;
    }

  // Direct memory access of the vector.
  typename dealii::Vector<double>::pointer p = a.data();
  for (unsigned int i = 0; i < N; i++)
    {
      std::cout << *(p + i) << " ";
    }
  std::cout << std::endl;

  // Check the memory consumption.
  std::cout << "Memory consumption: " << a.memory_consumption() << std::endl;
  a.reinit(200);
  std::cout << "Memory consumption after reinit: " << a.memory_consumption()
            << std::endl;

  return 0;
}
