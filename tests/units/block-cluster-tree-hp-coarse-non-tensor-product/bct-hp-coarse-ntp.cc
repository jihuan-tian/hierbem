// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file bct-hp-coarse-ntp.cc
 * \brief Test the construction of a \f$\mathcal{H}^p\f$ block cluster tree
 * using the coarse non-tensor product partition.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-06-22
 */

#include <fstream>
#include <iostream>

#include "cluster_tree/block_cluster_tree.h"
#include "cluster_tree/cluster_tree.h"

using namespace HierBEM;

int
main()
{
  {
    const unsigned int                   p = 2;
    const unsigned int                   n = std::pow(2, p);
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = (i + 1);
      }

    const unsigned int n_min = 1;

    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_coarse_non_tensor_product();

    std::cout << "=== Block cluster tree ===\n"
              << block_cluster_tree << std::endl;

    /**
     * Print the whole block cluster tree leaf set.
     */
    std::ofstream out("bct1.dat");
    block_cluster_tree.write_leaf_set(out);
  }

  /**
   * Generate a large structure.
   */
  {
    const unsigned int                   p = 6;
    const unsigned int                   n = std::pow(2, p);
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = (i + 1);
      }

    const unsigned int n_min = 1;

    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_coarse_non_tensor_product();

    std::cout << "=== Block cluster tree ===\n"
              << block_cluster_tree << std::endl;

    /**
     * Print the whole block cluster tree leaf set.
     */
    std::ofstream out("bct2.dat");
    block_cluster_tree.write_leaf_set(out);
  }

  {
    const unsigned int                   p = 2;
    const unsigned int                   n = std::pow(2, p);
    std::vector<types::global_dof_index> index_set(n);

    for (unsigned int i = 0; i < n; i++)
      {
        index_set.at(i) = (i + 1);
      }

    const unsigned int n_min = 2;

    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_coarse_non_tensor_product();

    std::cout << "=== Block cluster tree ===\n"
              << block_cluster_tree << std::endl;

    /**
     * Print the whole block cluster tree leaf set.
     */
    std::ofstream out("bct3.dat");
    block_cluster_tree.write_leaf_set(out);
  }

  {
    std::vector<types::global_dof_index> index_set(5);

    for (unsigned int i = 0; i < 5; i++)
      {
        index_set.at(i) = (i + 1);
      }

    const unsigned int n_min = 1;

    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_coarse_non_tensor_product();

    std::cout << "=== Block cluster tree ===\n"
              << block_cluster_tree << std::endl;

    /**
     * Print the whole block cluster tree leaf set.
     */
    std::ofstream out("bct4.dat");
    block_cluster_tree.write_leaf_set(out);
  }

  {
    std::vector<types::global_dof_index> index_set(5);

    for (unsigned int i = 0; i < 5; i++)
      {
        index_set.at(i) = (i + 1);
      }

    const unsigned int n_min = 2;

    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_coarse_non_tensor_product();

    std::cout << "=== Block cluster tree ===\n"
              << block_cluster_tree << std::endl;

    /**
     * Print the whole block cluster tree leaf set.
     */
    std::ofstream out("bct5.dat");
    block_cluster_tree.write_leaf_set(out);
  }

  return 0;
}
