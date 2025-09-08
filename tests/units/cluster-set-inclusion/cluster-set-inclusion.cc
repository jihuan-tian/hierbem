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
 * \file cluster-set-inclusion.cc
 * \brief Verify set inclusion operation for cluster index sets.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-21
 */

#include <iostream>

#include "cluster_tree/cluster.h"

int
main()
{
  std::vector<types::global_dof_index> index_set1{1, 2, 3};
  std::vector<types::global_dof_index> index_set2{1, 2};

  Cluster<3, double> cluster1(index_set1);
  Cluster<3, double> cluster2(index_set2);
  Cluster<3, double> cluster3(index_set1);

  std::cout << std::boolalpha << cluster2.is_subset(cluster1) << "\n"
            << cluster2.is_proper_subset(cluster1) << "\n"
            << cluster3.is_subset(cluster1) << "\n"
            << cluster3.is_proper_subset(cluster1) << "\n"
            << cluster1.is_superset(cluster2) << "\n"
            << cluster1.is_proper_superset(cluster2) << "\n"
            << cluster1.is_superset(cluster3) << "\n"
            << cluster1.is_proper_superset(cluster3) << std::endl;

  return 0;
}
