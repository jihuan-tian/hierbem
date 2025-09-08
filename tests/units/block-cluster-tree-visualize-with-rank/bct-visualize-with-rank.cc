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
 * \file bct-visualize-with-rank.cc
 * \brief Visualize the block cluster tree structure with matrix block's rank
 * calculated.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-04
 */

#include <fstream>
#include <iostream>

#include "cluster_tree/block_cluster_tree.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace HierBEM;

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

  const unsigned int n_min = 1;

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

  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter * std::sin(counter);
      counter += 1.0;
    }

  std::ofstream out("bct.dat");
  block_cluster_tree.write_leaf_set(out, M);
  out.close();

  // Print out the leaf set for comparison.
  block_cluster_tree.write_leaf_set(std::cout, M);

  return 0;
}
