/**
 * \file lapack-matrix-agglomeration-interwoven-indices.cc
 * \brief Verify the agglomeration of four full submatrices into a larger full
 * matrix when the index sets of several child clusters are interwoven together
 * into the index set of the parent cluster.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-07-31
 */

#include <deal.II/base/types.h>

#include <iostream>

#include "generic_functors.h"
#include "lapack_full_matrix_ext.h"

int
main()
{
  /**
   * Generate submatrices \p M11, \p M12, \p M21, \p M22 with their row and
   * column index sets for agglomeration.
   */
  std::vector<double> M11_values{1, 2, 3, 4, 5, 6};
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
   * Output the four submatrices.
   */
  M11.print_formatted_to_mat(std::cout, "M11", 5, false, 10, "0");
  M12.print_formatted_to_mat(std::cout, "M12", 5, false, 10, "0");
  M21.print_formatted_to_mat(std::cout, "M21", 5, false, 10, "0");
  M22.print_formatted_to_mat(std::cout, "M22", 5, false, 10, "0");

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
  /**
   * Output the agglomerated matrix \p M.
   */
  M.print_formatted_to_mat(std::cout, "M", 5, false, 10, "0");

  return 0;
}
