/**
 * \file hmatrix-add-formatted.cc
 * \brief Verify formatted addition of two \hmatrices.
 * \ingroup hierarchical_matrices testers
 * \author Jihuan Tian
 * \date 2021-07-03
 */

#include <cmath>
#include <iostream>

#include "hmatrix/hmatrix.h"

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
  const unsigned int fixed_rank_k = 1;

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
  M1.print_formatted_to_mat(std::cout, "M1");

  LAPACKFullMatrixExt<double> M2(n, n);
  counter = 1.0;
  for (auto it = M2.begin(); it != M2.end(); it++)
    {
      (*it) = counter * std::cos(counter);
      counter += 1.0;
    }
  M2.print_formatted_to_mat(std::cout, "M2");

  /**
   * Create the two H-matrices \p H1 and \p H2 from \p M1 and \p M2.
   */
  HMatrix<3, double> H1(block_cluster_tree, M1, fixed_rank_k);
  HMatrix<3, double> H2(block_cluster_tree, M2, fixed_rank_k);

  LAPACKFullMatrixExt<double> H1_full, H2_full, H1_add_H2_full;

  /**
   * Get the full matrix representation of \p H1 and \p H2.
   */
  H1.convertToFullMatrix(H1_full);
  H2.convertToFullMatrix(H2_full);

  /**
   * Add the full matrix versions of \p H1 and \p H2.
   */
  H1_full.add(H1_add_H2_full, H2_full);
  H1_add_H2_full.print_formatted_to_mat(
    std::cout, "H1_add_H2_full", 10, false, 1, "0");

  /**
   * Add the two H-matrices \p H1 and \p H2.
   */
  HMatrix<3, double> H(block_cluster_tree, fixed_rank_k);

  H1.add(H, H2, 1);

  /**
   * Convert the result matrix into a full matrix.
   */
  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 10, false, 1, "0");

  return 0;
}
