/**
 * \file bct-extend-to-finer-partition.cc
 * \brief Verify extend a block cluster tree to a given finer partition.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-07-23
 */

#include <fstream>
#include <iostream>

#include "block_cluster_tree.h"

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
   * Create two block cluster trees. One has the fine non-tensor product
   * partition and the other has the coarse non-tensor product partition.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The minimum cluster size \p n_min associated the cluster trees used
   * for building the block cluster tree might be different from that of the
   * block cluster tree. Usually, \p n_min for the cluster tree is set to 1,
   * which allows block cluster trees of different depths can be built from
   * it.</dd>
   * </dl>
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
   * Make a copy of the first tree and then extend it be finer than the second
   * tree.
   */
  BlockClusterTree<3, double> bct1_ext(bct1);
  bct1_ext.extend_finer_than_partition(bct2.get_leaf_set());
  std::ofstream out3("bct1_ext.dat");
  bct1_ext.write_leaf_set(out3);
  out3.close();

  /**
   * Make copy of the second tree and then extend it to the finer partition
   * which is obtained from above by extending the first tree.
   */
  BlockClusterTree<3, double> bct2_ext(bct2);
  bct2_ext.extend_to_finer_partition(bct1_ext.get_leaf_set());
  std::ofstream out4("bct2_ext.dat");
  bct2_ext.write_leaf_set(out4);
  out4.close();

  return 0;
}
