/**
 * \file hmatrix-bct-struct-with-rank.cc
 * \brief Visualize the block cluster tree structure of an H-matrix with
 * displayed rank.
 * \ingroup hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-07-04
 */

#include <fstream>
#include <iostream>
#include <string>

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

  const unsigned int n_min = 2;

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
   * Create the full matrix.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter * std::sin(counter);
      counter += 1.0;
    }

  /**
   * Convert the full matrix to H-matrix with specified rank. Then we convert it
   * back to a full matrix for comparison with the original matrix.
   */
  for (unsigned int k = 1; k <= 10; k++)
    {
      HMatrix<3, double> H(block_cluster_tree, M, k);
      
      // Print the leaf sets to console.
      std::cout << "* rank=" << k << "\n";
      H.write_leaf_set(std::cout, 1e-12);
      
      // Save the leaf sets to files.
      std::ofstream      output(std::string("bct-struct-with-rank=") +
                           std::to_string(k) + std::string(".dat"));
      H.write_leaf_set(output, 1e-12);
      output.close();
    }

  return 0;
}
