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
 * @file cudafullmatrix-mmult.cu
 * @brief Verify matrix-matrix multiplication.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "linear_algebra/cu_fullmatrix.hcu"

using namespace std;
using namespace HierBEM::CUDAWrappers;

int
main()
{
  double A_data[6]{1, 3, 5, 7, 9, 10};
  double B_data[9]{2, 8, 9, 7, 1, 3, 11, 20, 13};
  double C_data[6];
  double C_adding_data[6]{1, 1, 1, 2, 2, 2};

  CUDAFullMatrix<double> A(A_data, 2, 3);
  CUDAFullMatrix<double> B(B_data, 3, 3);
  CUDAFullMatrix<double> C(C_data, 2, 3);
  CUDAFullMatrix<double> C_adding(C_adding_data, 2, 3);

  A.mmult(C, B);
  A.mmult(C_adding, B, true);

  cout << "C=\n";
  C.print(false, false);

  cout << "C_adding=\n";
  C_adding.print(false, false);

  return 0;
}
