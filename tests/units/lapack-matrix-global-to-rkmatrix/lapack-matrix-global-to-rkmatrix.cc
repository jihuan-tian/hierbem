// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file lapack-matrix-global-to-rkmatrix.cc
 * \brief Verify the restriction of a global full matrix to a rank-k submatrix.
 * \ingroup rkmatrices test_cases
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  /**
   * Create a full matrix with data.
   */
  const unsigned int          n = 20;
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }
  M.print_formatted_to_mat(std::cout, "M");

  std::array<types::global_dof_index, 2> tau{7, 11};
  std::array<types::global_dof_index, 2> sigma{9, 13};

  RkMatrix<double> rkmat_no_trunc(tau, sigma, M);
  rkmat_no_trunc.print_formatted_to_mat(
    std::cout, "rkmat_no_trunc", 8, false, 16, "0");
  RkMatrix<double> rk1mat(tau, sigma, 1, M);
  rk1mat.print_formatted_to_mat(std::cout, "rk1mat", 8, false, 16, "0");

  return 0;
}
