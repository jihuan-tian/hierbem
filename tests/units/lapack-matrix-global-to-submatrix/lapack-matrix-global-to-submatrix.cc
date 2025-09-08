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
 * \file lapack-matrix-global-to-submatrix.cc
 * \brief Verify the restriction of a global full matrix to sub full matrix.
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <deal.II/base/types.h>

#include <iostream>

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
  std::array<types::global_dof_index, 2> sigma{3, 7};
  LAPACKFullMatrixExt<double>            M_b(tau, sigma, M);

  M_b.print_formatted_to_mat(std::cout, "M_b");

  return 0;
}
