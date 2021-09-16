/**
 * \file cluster-tree-copy-constructor.cc
 * \brief Verify the copy constructor of cluster tree.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-21
 */

#include <iostream>

#include "cluster_tree.h"

int
main()
{
  const unsigned int                   n = 32;
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
  ClusterTree<3> cluster_tree_orig(index_set, n_min);
  cluster_tree_orig.partition();

  ClusterTree<3> cluster_tree_copy(cluster_tree_orig);

  std::cout << "=== Original cluster tree ===\n"
            << cluster_tree_orig << std::endl;

  std::cout << "=== Copied cluster tree ===\n"
            << cluster_tree_copy << std::endl;

  return 0;
}
