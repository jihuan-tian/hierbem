// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file full-matrix-wrapper-vmult.cu
 * @brief Verify matrix-vector multiplication.
 *
 * @ingroup test_cases linalg
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "platform_shared/full_matrix_wrapper.h"
#include "platform_shared/vector_wrapper.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  double A_data[9]{2, 8, 9, 7, 1, 3, 11, 20, 13};
  double v_data[3]{7, 3, 10};
  double w_data[3];
  double w_adding_data[3]{1, 2, 3};

  PlatformShared::FullMatrixWrapper<double> A(A_data, 3, 3);
  PlatformShared::VectorWrapper<double>     v(v_data, 3);
  PlatformShared::VectorWrapper<double>     w(w_data, 3);
  PlatformShared::VectorWrapper<double>     w_adding(w_adding_data, 3);

  A.vmult(w, v);
  A.vmult(w_adding, v, true);

  cout << "w=\n";
  w.print(false);

  cout << "w_adding=\n";
  w_adding.print(false);

  return 0;
}
