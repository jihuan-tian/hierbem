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
 * \file rkmatrix-agglomeration-of-two-submatrices-rank0.cc
 * \brief Verify the agglomeration of two rank-k submatrices which have been
 * obtained from horizontal or vertical splitting. One of the submatrices has
 * rank 0.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> M1_values{1, 2, 3, 4, 5, 6};
  std::vector<double> M2_values{0, 0, 0, 0};
  std::vector<double> M3_values{0, 0, 0};

  LAPACKFullMatrixExt<double> M1, M2, M3;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, M1_values, M1);
  LAPACKFullMatrixExt<double>::Reshape(2, 2, M2_values, M2);
  LAPACKFullMatrixExt<double>::Reshape(1, 3, M3_values, M3);

  LAPACKFullMatrixExt<double> M_agglomerated1(M1, M2, false);
  LAPACKFullMatrixExt<double> M_agglomerated2(M1, M3, true);
  M_agglomerated1.print_formatted_to_mat(
    std::cout, "M_agglomerated1", 5, false, 10, "0");
  M_agglomerated2.print_formatted_to_mat(
    std::cout, "M_agglomerated2", 5, false, 10, "0");

  /**
   * Create rank-k matrices from the full matrices.
   */
  const unsigned int fixed_rank = 2;
  RkMatrix<double>   M1_rk(fixed_rank, M1);
  RkMatrix<double>   M2_rk(fixed_rank, M2);
  RkMatrix<double>   M3_rk(fixed_rank, M3);

  RkMatrix<double> M_agglomerated1_rk(fixed_rank, M1_rk, M2_rk, false);
  RkMatrix<double> M_agglomerated2_rk(fixed_rank, M1_rk, M3_rk, true);

  M_agglomerated1_rk.print_formatted_to_mat(
    std::cout, "M_agglomerated1_rk", 8, false, 16, "0");
  M_agglomerated2_rk.print_formatted_to_mat(
    std::cout, "M_agglomerated2_rk", 8, false, 16, "0");

  return 0;
}
