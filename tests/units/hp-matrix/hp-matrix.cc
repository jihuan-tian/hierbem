/**
 * \file hp-matrix.cc
 * \brief Test the \f$H^p\f$ matrix.
 *
 * \ingroup test_cases hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-06-23
 */

#include "hmatrix/hmatrix.h"

using namespace HierBEM;

int
main()
{
  const unsigned int                   p = 7;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min        = 1;
  const unsigned int fixed_rank_k = 1;

  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create a full matrix with data.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }

  /**
   * Create a rank-1 HMatrix.
   */
  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);
  hmat.print_formatted(std::cout, 5, false, 10, "0");

  /**
   * Convert the rank-1 HMatrix back to a full matrix.
   */
  LAPACKFullMatrixExt<double> M_tilde;
  hmat.convertToFullMatrix(M_tilde);
  std::cout << "=== HMatrix converted back to full matrix ===\n";
  M_tilde.print_formatted(std::cout, 5, false, 10, "0");

  return 0;
}
