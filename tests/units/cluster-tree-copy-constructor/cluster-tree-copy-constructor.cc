// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file cluster-tree-copy-constructor.cc
 * \brief Verify the copy constructor of cluster tree.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-21
 */

#include <iostream>

#include "cluster_tree/cluster_tree.h"

using namespace HierBEM;

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
