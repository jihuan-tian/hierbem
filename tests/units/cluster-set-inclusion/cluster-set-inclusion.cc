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
