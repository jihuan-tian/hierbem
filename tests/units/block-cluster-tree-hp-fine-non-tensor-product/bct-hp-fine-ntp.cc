/**
 * \file bct-hp-fine-ntp.cc
 * \brief
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-06-23
 */

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
    block_cluster_tree.partition_fine_non_tensor_product();

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
    const unsigned int                   p = 7;
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
    block_cluster_tree.partition_fine_non_tensor_product();

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
    block_cluster_tree.partition_fine_non_tensor_product();

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
    block_cluster_tree.partition_fine_non_tensor_product();

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
    block_cluster_tree.partition_fine_non_tensor_product();

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
