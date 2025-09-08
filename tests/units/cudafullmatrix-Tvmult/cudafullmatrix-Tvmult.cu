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
 * @file cudafullmatrix-Tvmult.cu
 * @brief Verify transposed matrix-vector multiplication.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "linear_algebra/cu_fullmatrix.hcu"
#include "linear_algebra/cu_vector.hcu"

using namespace std;
using namespace HierBEM::CUDAWrappers;

int
main()
{
  double A_data[12]{2, 8, 9, 7, 1, 3, 11, 20, 13, 20, 30, 10};
  double v_data[3]{7, 3, 10};
  double w_data[4];
  double w_adding_data[4]{1, 2, 3, 20};

  CUDAFullMatrix<double> A(A_data, 3, 4);
  CUDAVector<double>     v(v_data, 3);
  CUDAVector<double>     w(w_data, 4);
  CUDAVector<double>     w_adding(w_adding_data, 4);

  A.Tvmult(w, v);
  A.Tvmult(w_adding, v, true);

  cout << "w=\n";
  w.print(false);

  cout << "w_adding=\n";
  w_adding.print(false);

  return 0;
}
