/**
 * \file bct-copy-constructor.cc
 * \brief Verify the copy constructor of block cluster tree.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-21
 */

#include <fstream>
#include <iostream>

#include "block_cluster_tree.h"
#include "debug_tools.h"

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
                                                  n_min);
  block_cluster_tree1.partition_fine_non_tensor_product();
  std::cout << "=== BCT1 ===\n" << block_cluster_tree1;
  std::ofstream out1("bct1.dat");
  block_cluster_tree1.write_leaf_set(out1);
  out1.close();

  /**
   * Deep copy
   */
  BlockClusterTree<3, double> block_cluster_tree2(block_cluster_tree1);
  std::cout << "=== BCT2 ===\n" << block_cluster_tree2;
  std::ofstream out2("bct2.dat");
  block_cluster_tree2.write_leaf_set(out2);
  out2.close();

  /**
   * Shallow copy
   */
  BlockClusterTree<3, double> block_cluster_tree3(
    std::move(block_cluster_tree1));
  std::cout << "=== BCT3 ===\n" << block_cluster_tree3;
  std::ofstream out3("bct3.dat");
  block_cluster_tree3.write_leaf_set(out3);
  out3.close();

  /**
   * After shallow copy, try to print BCT1.
   */
  std::cout << "=== BCT1 after shallow copy ===\n" << block_cluster_tree1;

  return 0;
}
