/**
 * \file hmatrix-write-leaf-set-by-iteration.cu
 * \brief Verify the method for write out leaf set by iteration over the
 * constructed leaf set instead of recursion. By default, the traversal follows
 * the Z curve.
 *
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <iostream>

#include "hmatrix.h"

using namespace HierBEM;

int
main()
{
  /**
   * Create the global index set.
   */
  const unsigned int                   p = 5;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Construct the cluster tree.
   */
  const unsigned int n_min = 1;
  ClusterTree<3>     cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Construct the block cluster tree.
   */
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
  const unsigned int fixed_rank_k = 1;
  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);

  /**
   * Write out the leaf nodes by iteration.
   */
  hmat.write_leaf_set_by_iteration(std::cout);

  return 0;
}
