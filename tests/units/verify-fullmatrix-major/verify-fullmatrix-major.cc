// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-fullmatrix-major.cc
 * @brief Verify the major of the internal data in @p FullMatrix.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-01-30
 */

#include <deal.II/lac/full_matrix.h>

#include <iostream>

using namespace dealii;
using namespace std;

int
main()
{
  double values[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  FullMatrix<double> A(3, 5, values);

  cout << "A(0,2)=" << A(0, 2) << endl;
  cout << "A(2,0)=" << A(2, 0) << endl;

  double *start_ptr = &(A.begin()->value());
  for (unsigned int i = 0; i < 15; i++)
    {
      cout << *(start_ptr + i) << endl;
    }

  return 0;
}
