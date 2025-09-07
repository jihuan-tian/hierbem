/**
 * \file rkmatrix-agglomeration-of-two-submatrices-interwoven-indices-rank0.cc
 * \brief Verify the agglomeration of two rank-k submatrices which have been
 * obtained from horizontal or vertical splitting. The index sets of several
 * child clusters are interwoven together into the index set of the parent
 * cluster. One of the submatrices has rank zero.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/generic_functors.h"

int
main()
{
  std::vector<double> M1_values{1, 2, 3, 4, 5, 6};
  std::vector<double> M2_values{0, 0, 0, 0};
  std::vector<double> M3_values{0, 0, 0};

  LAPACKFullMatrixExt<double> M1, M2, M3;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, M1_values, M1);
  std::vector<dealii::types::global_dof_index> M1_tau_index_set{3, 7};
  std::vector<dealii::types::global_dof_index> M1_sigma_index_set{2, 8, 13};
  M1.print_formatted_to_mat(std::cout, "M1", 5, false, 10, "0");

  LAPACKFullMatrixExt<double>::Reshape(2, 2, M2_values, M2);
  std::vector<dealii::types::global_dof_index> M2_tau_index_set{3, 7};
  std::vector<dealii::types::global_dof_index> M2_sigma_index_set{6, 10};
  M2.print_formatted_to_mat(std::cout, "M2", 5, false, 10, "0");

  LAPACKFullMatrixExt<double>::Reshape(1, 3, M3_values, M3);
  std::vector<dealii::types::global_dof_index> M3_tau_index_set{9};
  std::vector<dealii::types::global_dof_index> M3_sigma_index_set{2, 8, 13};
  M3.print_formatted_to_mat(std::cout, "M3", 5, false, 10, "0");

  /**
   * Define the row and column index sets of the large matrix \p
   * M_agglomerated1 which is obtained from vertically split submatrices.
   */
  std::vector<dealii::types::global_dof_index> M_agglomerated1_tau_index_set{7,
                                                                             3};
  std::vector<dealii::types::global_dof_index> M_agglomerated1_sigma_index_set{
    2, 6, 8, 10, 13};

  /**
   * Define the row and column index sets of the large matrix \p
   * M_agglomerated2 which is obtained from horizontally split submatrices.
   */
  std::vector<dealii::types::global_dof_index> M_agglomerated2_tau_index_set{3,
                                                                             9,
                                                                             7};
  std::vector<dealii::types::global_dof_index> M_agglomerated2_sigma_index_set{
    8, 13, 2};

  /**
   * Build the global to local indices maps for the target agglomerated
   * matrices.
   */
  std::map<types::global_dof_index, size_t>
    row_index_global_to_local_map_for_M_agglomerated1;
  std::map<types::global_dof_index, size_t>
    col_index_global_to_local_map_for_M_agglomerated1;
  std::map<types::global_dof_index, size_t>
    row_index_global_to_local_map_for_M_agglomerated2;
  std::map<types::global_dof_index, size_t>
    col_index_global_to_local_map_for_M_agglomerated2;

  build_index_set_global_to_local_map(
    M_agglomerated1_tau_index_set,
    row_index_global_to_local_map_for_M_agglomerated1);
  build_index_set_global_to_local_map(
    M_agglomerated1_sigma_index_set,
    col_index_global_to_local_map_for_M_agglomerated1);
  build_index_set_global_to_local_map(
    M_agglomerated2_tau_index_set,
    row_index_global_to_local_map_for_M_agglomerated2);
  build_index_set_global_to_local_map(
    M_agglomerated2_sigma_index_set,
    col_index_global_to_local_map_for_M_agglomerated2);

  /**
   * Agglomeration of full submatrices, we should have
   * <code>
   * 2  0  4  0  6
   * 1  0  3  0  5
   * </code>
   */
  LAPACKFullMatrixExt<double> M_agglomerated1(
    row_index_global_to_local_map_for_M_agglomerated1,
    col_index_global_to_local_map_for_M_agglomerated1,
    M1,
    M1_tau_index_set,
    M1_sigma_index_set,
    M2,
    M2_tau_index_set,
    M2_sigma_index_set,
    false);

  /**
   * Agglomeration of full submatrices, we should have
   * <code>
   * 3   5   1
   * 0   0   0
   * 4   6   2
   * </code>
   */
  LAPACKFullMatrixExt<double> M_agglomerated2(
    row_index_global_to_local_map_for_M_agglomerated2,
    col_index_global_to_local_map_for_M_agglomerated2,
    M1,
    M1_tau_index_set,
    M1_sigma_index_set,
    M3,
    M3_tau_index_set,
    M3_sigma_index_set,
    true);
  M_agglomerated1.print_formatted_to_mat(
    std::cout, "M_agglomerated1", 5, false, 10, "0");
  M_agglomerated2.print_formatted_to_mat(
    std::cout, "M_agglomerated2", 5, false, 10, "0");

  /**
   * Create corresponding rank-k submatrices.
   */
  const unsigned int fixed_rank = 2;
  RkMatrix<double>   M1_rk(fixed_rank, M1);
  RkMatrix<double>   M2_rk(fixed_rank, M2);
  RkMatrix<double>   M3_rk(fixed_rank, M3);

  /**
   * Agglomeration of rank-k submatrices.
   */
  RkMatrix<double> M_agglomerated1_rk(
    fixed_rank,
    row_index_global_to_local_map_for_M_agglomerated1,
    col_index_global_to_local_map_for_M_agglomerated1,
    M1_rk,
    M1_tau_index_set,
    M1_sigma_index_set,
    M2_rk,
    M2_tau_index_set,
    M2_sigma_index_set,
    false);

  RkMatrix<double> M_agglomerated2_rk(
    fixed_rank,
    row_index_global_to_local_map_for_M_agglomerated2,
    col_index_global_to_local_map_for_M_agglomerated2,
    M1_rk,
    M1_tau_index_set,
    M1_sigma_index_set,
    M3_rk,
    M3_tau_index_set,
    M3_sigma_index_set,
    true);

  M_agglomerated1_rk.print_formatted_to_mat(
    std::cout, "M_agglomerated1_rk", 8, false, 16, "0");
  M_agglomerated2_rk.print_formatted_to_mat(
    std::cout, "M_agglomerated2_rk", 8, false, 16, "0");

  return 0;
}
