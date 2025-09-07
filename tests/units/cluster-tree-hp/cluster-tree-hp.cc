/**
 * \file cluster-tree-hp.cc
 * \brief Test the construction of a \f$\mathcal{H}^p\f$ cluster tree.
 * \ingroup hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-06-21
 */

#include <deal.II/base/logstream.h>

#include "cluster_tree/cluster_tree.h"

using namespace HierBEM;
using namespace dealii;

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

    /**
     * Print the whole cluster tree.
     */
    deallog << "=== Cluster tree ===\n";
    deallog << cluster_tree << std::endl;
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

    /**
     * Print the whole cluster tree.
     */
    deallog << "=== Cluster tree ===\n";
    deallog << cluster_tree << std::endl;
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

    /**
     * Print the whole cluster tree.
     */
    deallog << "=== Cluster tree ===\n";
    deallog << cluster_tree << std::endl;
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

    /**
     * Print the whole cluster tree.
     */
    deallog << "=== Cluster tree ===\n";
    deallog << cluster_tree << std::endl;
  }

  return 0;
}
