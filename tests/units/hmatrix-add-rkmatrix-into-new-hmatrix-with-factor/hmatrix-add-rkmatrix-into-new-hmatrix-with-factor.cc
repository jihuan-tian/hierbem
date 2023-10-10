/**
 * \file hmatrix-add-rkmatrix-into-new-hmatrix-with-factor.cc
 *
 * \brief Verify the addition of a rank-k matrix multiplied by a factor and an
 * \hmatrix into a new \hmatrix.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-10-05
 */

#include <cmath>
#include <iostream>

#include "hmatrix.h"

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
  const unsigned int fixed_rank_k = 2;

  /**
   * Generate cluster tree.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree via fine structured non-tensor product
   * partition.
   */
  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create two full matrices as the source data.
   */
  LAPACKFullMatrixExt<double> M1(n, n);
  double                      counter = 1.0;
  for (auto it = M1.begin(); it != M1.end(); it++)
    {
      (*it) = counter * std::sin(counter);
      counter += 1.0;
    }
  M1.print_formatted_to_mat(std::cout, "M1", 16, false, 25, "0");

  LAPACKFullMatrixExt<double> M2(n, n);
  counter = 1.0;
  for (auto it = M2.begin(); it != M2.end(); it++)
    {
      (*it) = counter * std::cos(counter);
      counter += 1.0;
    }
  M2.print_formatted_to_mat(std::cout, "M2", 16, false, 25, "0");

  /**
   * Create the H-matrix from \p M1.
   */
  HMatrix<3, double> H(block_cluster_tree, M1, fixed_rank_k);
  /**
   * Create the rank-k matrix from \p M2.
   */
  RkMatrix<double> R(M2);
  /**
   * Create the result \hmatrix.
   */
  HMatrix<3, double> H_sum;

  LAPACKFullMatrixExt<double> H_full, H_sum_full;

  /**
   * Get the full matrix representation of \p H1.
   */
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 16, false, 25, "0");

  /**
   * Add the rank-k matrix into the \hmatrix.
   */
  double b = 3.5;
  H.add(H_sum, b, R, fixed_rank_k);

  /**
   * Convert the result matrix into a full matrix.
   */
  H_sum.convertToFullMatrix(H_sum_full);
  H_sum_full.print_formatted_to_mat(
    std::cout, "H_sum_full", 16, false, 25, "0");

  return 0;
}
