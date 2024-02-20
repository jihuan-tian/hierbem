/**
 * \file bct-subtree.cc
 * \brief Verify the construction of a block cluster subtree and check the
 * equality of its nodes with the original block cluster tree.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-19
 */

#include <fstream>
#include <iostream>

#include "block_cluster_tree.h"
#include "debug_tools.hcu"

using namespace HierBEM;

int
main()
{
  /**
   * Set the dimension of the H^p-matrix to be built.
   */
  const unsigned int p = 5;
  const unsigned int n = std::pow(2, p);

  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set[i] = i;
    }

  /**
   * Set the minimum cluster size.
   */
  const unsigned int n_min = 2;

  /**
   * Generate the cluster tree using cardinality based partition.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate the block cluster tree.
   */
  BlockClusterTree<3, double> block_cluster_tree1(cluster_tree,
                                                  cluster_tree,
                                                  2);
  BlockClusterTree<3, double> block_cluster_tree2(cluster_tree,
                                                  cluster_tree,
                                                  6);

  block_cluster_tree1.partition_fine_non_tensor_product();
  block_cluster_tree2.partition_fine_non_tensor_product();

  std::ofstream out1("bct1.dat");
  block_cluster_tree1.write_leaf_set(out1);
  out1.close();

  std::ofstream out2("bct2.dat");
  block_cluster_tree2.write_leaf_set(out2);
  out2.close();

  // Print out leaf sets for comparison.
  block_cluster_tree1.write_leaf_set(std::cout);
  block_cluster_tree2.write_leaf_set(std::cout);

  return 0;
}
