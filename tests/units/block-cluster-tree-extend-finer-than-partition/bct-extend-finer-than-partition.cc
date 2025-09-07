/**
 * \file bct-extend-finer-than-partition.cc
 * \brief Verify extend a block cluster tree to be finer than a given partition.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-22
 */

#include <fstream>
#include <iostream>

#include "cluster_tree/block_cluster_tree.h"

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
  const unsigned int n_min = 1;

  /**
   * Generate the cluster tree using cardinality based partition.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Create two block cluster trees. One has the fine non-tensor product, and
   * the other has the coarse non-tensor product partition.
   */
  BlockClusterTree<3, double> bct1(cluster_tree, cluster_tree, 8);
  BlockClusterTree<3, double> bct2(cluster_tree, cluster_tree, 4);

  bct1.partition_fine_non_tensor_product();
  bct2.partition_coarse_non_tensor_product();

  std::ofstream out1("bct1.dat");
  bct1.write_leaf_set(out1);
  out1.close();

  std::ofstream out2("bct2.dat");
  bct2.write_leaf_set(out2);
  out2.close();

  /**
   * Make a copy of the first tree then extend it.
   */
  BlockClusterTree<3, double> bct1_ext(bct1);
  bct1_ext.extend_finer_than_partition(bct2.get_leaf_set());
  std::ofstream out3("bct1_ext.dat");
  bct1_ext.write_leaf_set(out3);
  out3.close();

  // Print out the leaf sets for comparison.
  bct1.write_leaf_set(std::cout);
  bct2.write_leaf_set(std::cout);
  bct1_ext.write_leaf_set(std::cout);

  return 0;
}
