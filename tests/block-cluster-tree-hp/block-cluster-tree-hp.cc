/**
 * \file block-cluster-tree-hp.cc
 * \brief Test the construction of a \f$\mathcal{H}^p\f$ block cluster tree
 * using the tensor product partition.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-06-22
 */

#include <deal.II/base/logstream.h>

#include <fstream>

#include "block_cluster_tree.h"
#include "cluster_tree.h"

int
main()
{
  /**
   * Initialize deal.ii log stream.
   */
  deallog.pop();
  deallog.depth_console(2);

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
    block_cluster_tree.partition_tensor_product();

    /**
     * Print the whole block cluster tree.
     */
    deallog << "=== Block cluster tree ===\n";
    deallog << block_cluster_tree << std::endl;

    /**
     * Print the tree structure.
     */
    std::ofstream out("bct1.dat");
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
    block_cluster_tree.partition_tensor_product();

    /**
     * Print the whole block cluster tree.
     */
    deallog << "=== Block cluster tree ===\n";
    deallog << block_cluster_tree << std::endl;

    /**
     * Print the tree structure.
     */
    std::ofstream out("bct2.dat");
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
    block_cluster_tree.partition_tensor_product();

    /**
     * Print the whole block cluster tree.
     */
    deallog << "=== Block cluster tree ===\n";
    deallog << block_cluster_tree << std::endl;

    /**
     * Print the tree structure.
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

    const unsigned int n_min = 2;

    ClusterTree<3> cluster_tree(index_set, n_min);
    cluster_tree.partition();

    BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
    block_cluster_tree.partition_tensor_product();

    /**
     * Print the whole block cluster tree.
     */
    deallog << "=== Block cluster tree ===\n";
    deallog << block_cluster_tree << std::endl;

    /**
     * Print the tree structure.
     */
    std::ofstream out("bct4.dat");
    block_cluster_tree.write_leaf_set(out);
  }

  return 0;
}
