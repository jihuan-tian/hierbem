/**
 * \file rkmatrix-assemble-from-rkmatrix.cc
 * \brief Verify assemble a rank-k matrix into a larger rank-k matrix.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-09-27
 */

#include "hmatrix/rkmatrix.h"

#include <deal.II/base/types.h>

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/generic_functors.h"

int
main()
{
  std::vector<double> M12_values{7, 8, 9, 10};

  LAPACKFullMatrixExt<double> M12;
  LAPACKFullMatrixExt<double>::Reshape(2, 2, M12_values, M12);
  std::vector<dealii::types::global_dof_index> M12_tau_index_set{7, 11};
  std::vector<dealii::types::global_dof_index> M12_sigma_index_set{6, 10};

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
  std::vector<double>         M_values{1,  2,  3,  4,  5,  6,  7,  8,  9,
                               10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25};
  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(M_tau_index_set.size(),
                                       M_sigma_index_set.size(),
                                       M_values,
                                       M);
  M.fill(row_index_global_to_local_map_for_M,
         col_index_global_to_local_map_for_M,
         M12,
         M12_tau_index_set,
         M12_sigma_index_set,
         true);
  M12.print_formatted_to_mat(std::cout, "M12", 5, false, 10, "0");
  M.print_formatted_to_mat(std::cout, "M", 5, false, 10, "0");

  const unsigned int fixed_rank_k = 3;
  RkMatrix<double>   M12_rk(fixed_rank_k, M12);

  LAPACKFullMatrixExt<double> M_tilde;
  LAPACKFullMatrixExt<double>::Reshape(M_tau_index_set.size(),
                                       M_sigma_index_set.size(),
                                       M_values,
                                       M_tilde);
  RkMatrix<double> M_rk(fixed_rank_k, M_tilde);
  M_rk.assemble_from_rkmatrix(row_index_global_to_local_map_for_M,
                              col_index_global_to_local_map_for_M,
                              M12_rk,
                              M12_tau_index_set,
                              M12_sigma_index_set,
                              fixed_rank_k);

  /**
   * Output the matrices. The assembled matrix \p M should be
   * \f[
   * M = \begin{pmatrix}
   * 1 &  6 & 11 & 16 & 21 \\
   * 2 &  7 & 12 & 17 & 22 \\
   * 3 & 15 & 13 & 27 & 23 \\
   * 4 &  9 & 14 & 19 & 24 \\
   * 5 & 18 & 15 & 30 & 25
   * \end{pmatrix}
   * \f]
   */
  M12_rk.print_formatted_to_mat(std::cout, "M12_rk", 5, false, 10, "0");
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 5, false, 10, "0");

  return 0;
}
