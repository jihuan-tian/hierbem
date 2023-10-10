/**
 * \file hmatrix-truncate-to-fixed-rank.cc
 * \brief Verify the rank truncation of an \hmatrix.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-06-24
 */

#include "hmatrix.h"

int
main()
{
  const unsigned int p = 4;
  const unsigned int n = std::pow(2, p);

  /**
   * Create a full matrix as the data source.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }
  M.print_formatted_to_mat(std::cout, "M");

  /**
   * Create an \hmatrix of fixed rank 2 using the fine non-tensor product
   * partition. In practice, the rank may not be a constant but a block
   * dependent distribution or map.
   */

  std::vector<types::global_dof_index> index_set(n);
  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min        = 2;
  const unsigned int fixed_rank_k = 2;

  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);
  std::cout << "HMatrix rank-2:\n";
  hmat.print_formatted(std::cout, 5, false, 10, "0");

  /**
   * Truncate the HMatrix to fixed rank 1.
   */
  hmat.truncate_to_rank(1);
  std::cout << "HMatrix rank-1:\n";
  hmat.print_formatted(std::cout, 5, false, 10, "0");

  /**
   * Convert the HMatrix back to a full matrix.
   */
  LAPACKFullMatrixExt<double> M_prime;
  hmat.convertToFullMatrix(M_prime);
  M_prime.print_formatted_to_mat(std::cout, "M_prime");

  return 0;
}
