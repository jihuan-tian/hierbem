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
 * @file cudafullmatrix-invert2x2.cu
 * @brief Verify the calculation of the invert of a 2x2 matrix.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "linear_algebra/cu_fullmatrix.hcu"

using namespace HierBEM::CUDAWrappers;
using namespace std;

int
main()
{
  double data[4]{1, 2, 3, 4};
  double data_inv[4];

  CUDAFullMatrix<double> A(data, 2, 2);
  CUDAFullMatrix<double> A_inv(data_inv, 2, 2);

  A_inv.invert2x2(A);

  A_inv.print();

  return 0;
}
