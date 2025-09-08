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
 * \file rkmatrix-global-to-submatrix.cc
 * \brief Verify the restriction of a global rank-k matrix to a full submatrix.
 * \ingroup
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

  /**
   * Create a rank-k matrix from the full matrix.
   */
  RkMatrix<double> M_rk(2, M);
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 8, false, 16, "0");

  /**
   * Create a full submatrix matrix by restriction from the large rank-k matrix
   * on the block cluster \f$\tau \times \sigma\f$.
   */
  std::vector<types::global_dof_index> tau{2, 3, 4, 5, 7, 10, 18, 19};
  std::vector<types::global_dof_index> sigma{3, 4, 8, 9, 11, 13, 15, 16, 17};
  LAPACKFullMatrixExt<double>          M_b;
  M_rk.restrictToFullMatrix(tau, sigma, M_b);
  M_b.print_formatted_to_mat(std::cout, "M_b", 8, false, 16, "0");

  return 0;
}
