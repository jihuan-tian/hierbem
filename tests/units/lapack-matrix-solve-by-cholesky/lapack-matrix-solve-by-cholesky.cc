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
 * \file lapack-matrix-solve-by-cholesky.cc
 * \brief Verify solving a full matrix using Cholesky decomposition.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-22
 */

#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"
#include "utilities/read_octave_data.h"

int
main()
{
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");


  dealii::Vector<double> b({1, 3, 2, 4, 5, 7, 9, 6});
  print_vector_to_mat(std::cout, "b", b, false);

  /**
   * Compute Cholesky decomposition of the matrix.
   */
  M.set_property(LAPACKSupport::symmetric);
  M.compute_cholesky_factorization();

  /**
   * Solve the matrix.
   */
  M.solve(b);

  /**
   * Print the result vector, which is stored in \p b.
   */
  print_vector_to_mat(std::cout, "x", b, false);

  return 0;
}
