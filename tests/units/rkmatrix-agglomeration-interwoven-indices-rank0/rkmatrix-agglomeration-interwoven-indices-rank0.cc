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
 * \file rkmatrix-agglomeration-interwoven-indices-rank0.cc
 * \brief Verify the agglomeration of four rank-k submatrices into a larger
 * rank-k matrix when the index sets of several child clusters are interwoven
 * together into the index set of the parent cluster. Some of the submatrices
 * has rank zero.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <deal.II/base/types.h>

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/generic_functors.h"

int
main()
{
  // std::vector<double> M11_values{1, 2, 3, 4, 5, 6};
  std::vector<double> M11_values{0, 0, 0, 0, 0, 0};
  std::vector<double> M12_values{7, 8, 9, 10};
  std::vector<double> M21_values{11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<double> M22_values{20, 21, 22, 23, 24, 25};

  LAPACKFullMatrixExt<double> M11, M12, M21, M22;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, M11_values, M11);
  std::vector<dealii::types::global_dof_index> M11_tau_index_set{7, 11};
  std::vector<dealii::types::global_dof_index> M11_sigma_index_set{2, 8, 13};

  LAPACKFullMatrixExt<double>::Reshape(2, 2, M12_values, M12);
  std::vector<dealii::types::global_dof_index> M12_tau_index_set{7, 11};
  std::vector<dealii::types::global_dof_index> M12_sigma_index_set{6, 10};

  LAPACKFullMatrixExt<double>::Reshape(3, 3, M21_values, M21);
  std::vector<dealii::types::global_dof_index> M21_tau_index_set{1, 3, 9};
  std::vector<dealii::types::global_dof_index> M21_sigma_index_set{2, 8, 13};

  LAPACKFullMatrixExt<double>::Reshape(3, 2, M22_values, M22);
  std::vector<dealii::types::global_dof_index> M22_tau_index_set{1, 3, 9};
  std::vector<dealii::types::global_dof_index> M22_sigma_index_set{6, 10};

  /**
   * Define the row and column index sets of the large matrix \p M, which has
   * a dimension \f$5 \times 5\f$.
   */
  std::vector<dealii::types::global_dof_index> M_tau_index_set{1, 3, 7, 9, 11};
  std::vector<dealii::types::global_dof_index> M_sigma_index_set{
    2, 6, 8, 10, 13};

  /**
   * Build the global to local indices map for the matrix \p M.
   */
  std::map<types::global_dof_index, size_t> row_index_global_to_local_map_for_M;
  std::map<types::global_dof_index, size_t> col_index_global_to_local_map_for_M;
  build_index_set_global_to_local_map(M_tau_index_set,
                                      row_index_global_to_local_map_for_M);
  build_index_set_global_to_local_map(M_sigma_index_set,
                                      col_index_global_to_local_map_for_M);

  /**
   * Build the agglomerated full matrix which is to be compared with rank-k
   * matrix.
   */
  LAPACKFullMatrixExt<double> M(row_index_global_to_local_map_for_M,
                                col_index_global_to_local_map_for_M,
                                M11,
                                M11_tau_index_set,
                                M11_sigma_index_set,
                                M12,
                                M12_tau_index_set,
                                M12_sigma_index_set,
                                M21,
                                M21_tau_index_set,
                                M21_sigma_index_set,
                                M22,
                                M22_tau_index_set,
                                M22_sigma_index_set);

  M11.print_formatted_to_mat(std::cout, "M11", 15, false, 25, "0");
  M12.print_formatted_to_mat(std::cout, "M12", 15, false, 25, "0");
  M21.print_formatted_to_mat(std::cout, "M21", 15, false, 25, "0");
  M22.print_formatted_to_mat(std::cout, "M22", 15, false, 25, "0");
  M.print_formatted_to_mat(std::cout, "M", 15, false, 25, "0");

  const unsigned int fixed_rank_k = 4;

  RkMatrix<double> M11_rk(fixed_rank_k, M11);
  RkMatrix<double> M12_rk(fixed_rank_k, M12);
  RkMatrix<double> M21_rk(fixed_rank_k, M21);
  RkMatrix<double> M22_rk(fixed_rank_k, M22);

  RkMatrix<double> M_rk(fixed_rank_k,
                        row_index_global_to_local_map_for_M,
                        col_index_global_to_local_map_for_M,
                        M11_rk,
                        M11_tau_index_set,
                        M11_sigma_index_set,
                        M12_rk,
                        M12_tau_index_set,
                        M12_sigma_index_set,
                        M21_rk,
                        M21_tau_index_set,
                        M21_sigma_index_set,
                        M22_rk,
                        M22_tau_index_set,
                        M22_sigma_index_set,
                        1.5);

  /**
   * Output the matrices.
   */
  // M11_rk.print_formatted_to_mat(std::cout, "M11_rk", 15, false, 25, "0");
  M12_rk.print_formatted_to_mat(std::cout, "M12_rk", 15, false, 25, "0");
  M21_rk.print_formatted_to_mat(std::cout, "M21_rk", 15, false, 25, "0");
  M22_rk.print_formatted_to_mat(std::cout, "M22_rk", 15, false, 25, "0");
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 15, false, 25, "0");

  return 0;
}
