/**
 * \file hmatrix-hmatrix-Tmmult-level-conserving-symm-result-ntp.cc
 * \brief Verify the \hmat multiplication \f$M^T \cdot M\f$, hence the result
 * matrix is symmetric.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-11-28
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "hmatrix.h"

int
main()
{
  const unsigned int p = 4;
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
   * Generate block cluster tree via fine structured non-tensor product
   * partition.
   */
  const unsigned int          n_min_bct = 2;
  BlockClusterTree<3, double> bc_tree1(cluster_tree, cluster_tree, n_min_bct);
  bc_tree1.partition_fine_non_tensor_product();
  BlockClusterTree<3, double> bc_tree2(bc_tree1);
  BlockClusterTree<3, double> bc_tree3(bc_tree1);

  /**
   * Create full matrix as the source data.
   */
  LAPACKFullMatrixExt<double> M1(n, n);
  double                      counter = 1.0;
  for (auto it = M1.begin(); it != M1.end(); it++)
    {
      (*it) = std::sin(counter);
      counter += 1.0;
    }
  M1.print_formatted_to_mat(std::cout, "M1", 16, false, 25, "0");

  /**
   * Create the H-matrix \p H1 from \p M1.
   */
  HMatrix<3, double> H1(bc_tree1, M1, fixed_rank_k);

  /**
   * Create the empty symmetric result \hmatrix \p H.
   */
  HMatrix<3, double> H(bc_tree3.get_root(),
                       fixed_rank_k,
                       HMatrixSupport::Property::symmetric,
                       HMatrixSupport::BlockType::diagonal_block);
  /**
   * Get the full matrix representations of \p H1 as well as \f$H1 \cdot H1^T\f$.
   */
  LAPACKFullMatrixExt<double> H1_full, H1T_mult_H1_full;
  H1.convertToFullMatrix(H1_full);
  H1_full.print_formatted_to_mat(std::cout, "H1_full", 16, false, 25, "0");

  H1_full.Tmmult(H1T_mult_H1_full, H1_full);
  H1T_mult_H1_full.print_formatted_to_mat(
    std::cout, "H1T_mult_H1_full", 16, false, 25, "0");

  /**
   * Multiply the \hmatrices
   */
  H1.Tmmult_level_conserving(H, H1, fixed_rank_k, true);
  std::ofstream H_out("H_bct.dat");
  H.write_leaf_set_by_iteration(H_out);
  H_out.close();

  /**
   * Convert the result matrix into a full matrix for verification.
   */
  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 16, false, 25, "0");

  return 0;
}
