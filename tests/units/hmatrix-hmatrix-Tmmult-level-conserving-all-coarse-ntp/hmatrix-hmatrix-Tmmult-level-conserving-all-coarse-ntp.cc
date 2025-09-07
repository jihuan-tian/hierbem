/**
 * \file hmatrix-hmatrix-Tmmult-level-conserving-all-coarse-ntp.cc
 * \brief Verify the multiplication of two level-conserving
 * \f$\mathcal{H}\f$-matrices with the first operand transposed. Both operands
 * and the result matrices have the coarse non-tensor product partitions.
 *
 * \ingroup test_cases hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-11-28
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "hmatrix/hmatrix.h"

using namespace HierBEM;

int
main()
{
  const unsigned int p = 5;
  const unsigned int n = std::pow(2, p);

  /**
   * Generate index set.
   */
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min        = 1;
  unsigned int       fixed_rank_k = 2;

  /**
   * Generate cluster tree.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree via coarse structured non-tensor product
   * partition.
   */
  const unsigned int          n_min_bct = 2;
  BlockClusterTree<3, double> bc_tree1(cluster_tree, cluster_tree, n_min_bct);
  bc_tree1.partition_coarse_non_tensor_product();
  BlockClusterTree<3, double> bc_tree2(bc_tree1);
  BlockClusterTree<3, double> bc_tree3(bc_tree1);

  /**
   * Create two full matrices as the source data.
   */
  LAPACKFullMatrixExt<double> M1(n, n);
  double                      counter = 1.0;
  for (auto it = M1.begin(); it != M1.end(); it++)
    {
      (*it) = std::sin(counter);
      counter += 1.0;
    }
  M1.print_formatted_to_mat(std::cout, "M1", 16, false, 25, "0");

  LAPACKFullMatrixExt<double> M2(n, n);
  counter = 1.0;
  for (auto it = M2.begin(); it != M2.end(); it++)
    {
      (*it) = std::cos(counter);
      counter += 1.0;
    }
  M2.print_formatted_to_mat(std::cout, "M2", 16, false, 25, "0");

  /**
   * Create the two H-matrices \p H1 and \p H2 from \p M1 and \p M2.
   */
  HMatrix<3, double> H1(bc_tree1, M1, fixed_rank_k);
  HMatrix<3, double> H2(bc_tree2, M2, fixed_rank_k);
  std::ofstream      H1_out("H1_bct.dat");
  H1.write_leaf_set_by_iteration(H1_out);
  H1_out.close();
  std::ofstream H2_out("H2_bct.dat");
  H2.write_leaf_set_by_iteration(H2_out);
  H2_out.close();

  /**
   * Create the empty result \hmatrix \p H3.
   */
  HMatrix<3, double> H3(bc_tree3.get_root(), fixed_rank_k);

  /**
   * Create the empty result \hmatrix \p H4.
   */
  HMatrix<3, double> H4(bc_tree3.get_root(), fixed_rank_k);

  /**
   * Get the full matrix representations of \p H1 and \p H2 as well as their
   * product.
   */
  LAPACKFullMatrixExt<double> H1_full, H2_full, H1_mult_H2_full;
  H1.convertToFullMatrix(H1_full);
  H2.convertToFullMatrix(H2_full);
  H1_full.print_formatted_to_mat(std::cout, "H1_full", 16, false, 25, "0");
  H2_full.print_formatted_to_mat(std::cout, "H2_full", 16, false, 25, "0");

  H1_full.Tmmult(H1_mult_H2_full, H2_full);
  H1_mult_H2_full.print_formatted_to_mat(
    std::cout, "H1_mult_H2_full", 16, false, 25, "0");

  /**
   * Multiply the two H-matrices \p H1 and \p H2.
   */
  h_h_Tmmult_level_conserving(H3, H1, H2, fixed_rank_k);
  std::ofstream H3_out("H3_bct.dat");
  H3.write_leaf_set_by_iteration(H3_out);
  H3_out.close();

  H1.Tmmult_level_conserving(H4, 0.5, H2, fixed_rank_k, false);

  /**
   * Convert the result matrix into a full matrix for verification.
   */
  LAPACKFullMatrixExt<double> H3_full;
  H3.convertToFullMatrix(H3_full);
  H3_full.print_formatted_to_mat(std::cout, "H3_full", 16, false, 25, "0");

  LAPACKFullMatrixExt<double> H4_full;
  H4.convertToFullMatrix(H4_full);
  H4_full.print_formatted_to_mat(std::cout, "H4_full", 16, false, 25, "0");

  return 0;
}
