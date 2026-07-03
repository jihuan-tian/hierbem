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
 * @file full-matrix-wrapper-invert-by-gauss-elim.cu
 * @brief Verify matrix inversion by Gauss elimination.
 *
 * @ingroup test_cases linalg
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include "platform_shared/full_matrix_wrapper.h"

using namespace HierBEM;

int
main()
{
  constexpr unsigned int N = 8;
  double data[N * N]{-1., 5.,  2.,  -3., 6.,  1.,  -2., 4.,  2.,  -3., -4.,
                     1.,  -3., -1., 1.,  2.,  -2., 4.,  2.,  -1., 3.,  1.,
                     -1., 3.,  -3., 7.,  2.,  -3., 7.,  2.,  -2., 2.,  1.,
                     0.,  0.,  -1., 1.,  -4., 0.,  0.,  0.,  2.,  0.,  -2.,
                     3.,  -1., -1., 6.,  -2., 4.,  3.,  -2., 4.,  -1., -1.,
                     3.,  3.,  -4., -6., 1.,  -3., -3., 1.,  -2};
  double data_inv[N * N];

  PlatformShared::FullMatrixWrapper<double> A(data, N, N);
  PlatformShared::FullMatrixWrapper<double> A_inv(data_inv, N, N);

  A.invert_by_gauss_elim(A_inv);

  A_inv.print();

  return 0;
}
