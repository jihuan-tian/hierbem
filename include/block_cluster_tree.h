/**
 * \file block_cluster_tree.h
 * \brief Implementation of the class BlockClusterTree.
 * \ingroup hierarchical_matrices
 * \date 2021-04-20
 * \author Jihuan Tian
 */

#ifndef INCLUDE_BLOCK_CLUSTER_TREE_H_
#define INCLUDE_BLOCK_CLUSTER_TREE_H_

/**
 * \ingroup hierarchical_matrices
 * @{
 */

#include "block_cluster.h"
#include "cluster_tree.h"
#include "tree.h"

/**
 * \brief Class for block cluster tree.
 *
 * A block cluster tree is a quad-tree which holds a hierarchy of linked nodes
 * with the type TreeNode. Because a node in the block cluster tree has four
 * children, the template argument \p T required by \p TreeNode should be 4.
 *
 * At present, only a list of admissible or small block clusters at the deepest
 * level is constructed by applying partitioning from the root node, the tree
 * structure will not be generated.
 */
template <int spacedim, typename Number = double>
class BlockClusterTree
{
public:
  /**
   * Print a whole block cluster tree using recursion.
   * @param out
   * @param block_cluster_tree
   * @return
   */
  template <int spacedim1, typename Number1>
  friend std::ostream &
  operator<<(std::ostream &                              out,
             const BlockClusterTree<spacedim1, Number1> &block_cluster_tree);

  /**
   * Number of children in a block cluster tree.
   *
   * At present, only quad-tree is allowed.
   */
  static const unsigned int child_num = 4;

  /**
   * Data type of the tree node.
   */
  typedef TreeNode<BlockCluster<spacedim, Number>, child_num> node_value_type;
  typedef TreeNode<BlockCluster<spacedim, Number>, child_num>
    *node_pointer_type;
  typedef const TreeNode<BlockCluster<spacedim, Number>, child_num>
    *node_const_pointer_type;
  typedef TreeNode<BlockCluster<spacedim, Number>, child_num>
    &node_reference_type;
  typedef const TreeNode<BlockCluster<spacedim, Number>, child_num>
    &node_const_reference_type;

  /**
   * Data type of the data held by a tree node.
   */
  typedef BlockCluster<spacedim, Number>        data_value_type;
  typedef BlockCluster<spacedim, Number> *      data_pointer_type;
  typedef const BlockCluster<spacedim, Number> *data_const_pointer_type;
  typedef BlockCluster<spacedim, Number> &      data_reference_type;
  typedef const BlockCluster<spacedim, Number> &data_const_reference_type;

  /**
   * Default constructor, which initializes an empty quad-tree.
   */
  BlockClusterTree();

  /**
   * Destructor which recursively destroys every node in the block cluster tree.
   */
  ~BlockClusterTree();

  /**
   * Construct from two cluster trees.
   *
   * The Cartesian product of the two clusters in the root nodes of \f$T(I)\f$
   * and \f$T(J)\f$ becomes the data in the root node of the block cluster tree.
   */
  BlockClusterTree(const ClusterTree<spacedim, Number> &TI,
                   const ClusterTree<spacedim, Number> &TJ,
                   Number                               eta);

  /**
   * Perform a recursive partition by starting from the root node. The
   * evaluation of the admissibility condition has no mesh cell size correction.
   *
   * @param all_support_points All the support points.
   */
  void
  partition(const std::vector<Point<spacedim>> &all_support_points);

  /**
   * Perform a recursive partition by starting from the root node. The
   * evaluation of the admissibility condition has mesh cell size correction.
   *
   * @param all_support_points All the support points.
   */
  void
  partition(const std::vector<Point<spacedim>> &all_support_points,
            const std::vector<Number> &         cell_size_at_dofs);

  /**
   * Get the pointer to the root node of the block cluster tree.
   * @return
   */
  node_pointer_type
  get_root() const;

  /**
   * Get the reference to the block cluster list.
   */
  std::vector<data_value_type> &
  get_block_cluster_list();

  /**
   * Get the reference to the block cluster list (const version).
   */
  const std::vector<data_value_type> &
  get_block_cluster_list() const;

  /**
   * Get the tree depth.
   */
  unsigned int
  get_depth() const;

private:
  /**
   * Perform a recursive partition by starting from a block cluster node. The
   * evaluation of the admissibility condition has no mesh cell size correction.
   *
   * @param all_support_points All the support points.
   */
  std::vector<data_value_type>
  partition_from_block_cluster(
    data_const_reference_type           current_block_cluster,
    const std::vector<Point<spacedim>> &all_support_points);

  /**
   * Perform a recursive partition by starting from a block cluster node. The
   * evaluation of the admissibility condition has mesh cell size correction.
   *
   * @param all_support_points All the support points.
   */
  std::vector<data_value_type>
  partition_from_block_cluster(
    data_const_reference_type           current_block_cluster,
    const std::vector<Point<spacedim>> &all_support_points,
    const std::vector<Number> &         cell_size_at_dofs);

  node_pointer_type            root_node;
  std::vector<data_value_type> block_cluster_list;
  unsigned int                 n_min;
  Number                       eta;

  /**
   * Depth of the tree, which is the maximum level plus one.
   */
  unsigned int depth;
};

template <int spacedim, typename Number>
std::ostream &
operator<<(std::ostream &                            out,
           const BlockClusterTree<spacedim, Number> &block_cluster_tree)
{
  out << "* Tree depth: " << block_cluster_tree.depth << "\n";
  PrintTree(out, block_cluster_tree.root_node);

  return out;
}

template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree()
  : root_node(nullptr)
  , block_cluster_list(0)
  , n_min(2)
  , eta(1.0)
  , depth(0)
{}

template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  const ClusterTree<spacedim, Number> &TI,
  const ClusterTree<spacedim, Number> &TJ,
  Number                               eta)
  : root_node(nullptr)
  , block_cluster_list(0)
  , n_min(std::min(TI.get_n_min(), TJ.get_n_min()))
  , eta(eta)
  , depth(1)
{
  // Initialize the four null child pointers.
  std::array<node_pointer_type, child_num> empty_children_pointers{
    {nullptr, nullptr, nullptr, nullptr}};
  root_node = CreateTreeNode<data_value_type>(
    BlockCluster<spacedim, Number>(TI.get_root(), TJ.get_root()),
    0,
    empty_children_pointers);
}

template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::~BlockClusterTree()
{
  DeleteTree(root_node);
}

template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<Point<spacedim>> &all_support_points)
{
  block_cluster_list =
    partition_from_block_cluster(*(root_node->get_data_pointer()),
                                 all_support_points);

  // Update the tree depth.
  depth = calc_depth(root_node);
}

template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number> &         cell_size_at_dofs)
{
  block_cluster_list =
    partition_from_block_cluster(*(root_node->get_data_pointer()),
                                 all_support_points,
                                 cell_size_at_dofs);

  // Update the tree depth.
  depth = calc_depth(root_node);
}

template <int spacedim, typename Number>
typename BlockClusterTree<spacedim, Number>::node_pointer_type
BlockClusterTree<spacedim, Number>::get_root() const
{
  return root_node;
}

template <int spacedim, typename Number>
std::vector<typename BlockClusterTree<spacedim, Number>::data_value_type>
BlockClusterTree<spacedim, Number>::partition_from_block_cluster(
  data_const_reference_type           current_block_cluster,
  const std::vector<Point<spacedim>> &all_support_points)
{
  std::vector<data_value_type> local_block_cluster_list(0);

  if (current_block_cluster.is_admissible_or_small(eta,
                                                   all_support_points,
                                                   n_min))
    {
      local_block_cluster_list.push_back(current_block_cluster);
    }
  else
    {
      // Iterate over each child of the cluster node for \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster.get_tau_node()->get_child_pointer(i);
          // Iterate over each child of the cluster node for \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster.get_sigma_node()->get_child_pointer(j);

              /**
               * Make sure that the two clusters have the same level in their
               * respective cluster trees, i.e. level preserving property.
               */
              Assert(tau_son_node_pointer->get_level() ==
                       sigma_son_node_pointer->get_level(),
                     ExcDimensionMismatch(tau_son_node_pointer->get_level(),
                                          sigma_son_node_pointer->get_level()));
              deallog
                << "Block cluster cardinality: ["
                << tau_son_node_pointer->get_data_pointer()->get_cardinality()
                << ", "
                << sigma_son_node_pointer->get_data_pointer()->get_cardinality()
                << "]\n";

              /**
               * Create a new block cluster and recursively partition from
               * it.
               */
              BlockCluster<spacedim, Number> new_block_cluster(
                tau_son_node_pointer, sigma_son_node_pointer);

              std::vector<data_value_type> temp_block_cluster_list(
                partition_from_block_cluster(new_block_cluster,
                                             all_support_points));

              /**
               * Append the resulted block cluster list obtained from the
               * recursion to the local block cluster list.
               */
              for (const auto &block_cluster : temp_block_cluster_list)
                {
                  local_block_cluster_list.push_back(block_cluster);
                }
            }
        }
    }

  return local_block_cluster_list;
}

template <int spacedim, typename Number>
std::vector<typename BlockClusterTree<spacedim, Number>::data_value_type>
BlockClusterTree<spacedim, Number>::partition_from_block_cluster(
  data_const_reference_type           current_block_cluster,
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number> &         cell_size_at_dofs)
{
  std::vector<data_value_type> local_block_cluster_list(0);

  if (current_block_cluster.is_admissible_or_small(
        eta, all_support_points, cell_size_at_dofs, n_min))
    {
      local_block_cluster_list.push_back(current_block_cluster);
    }
  else
    {
      // Iterate over each child of the cluster node for \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster.get_tau_node()->get_child_pointer(i);
          // Iterate over each child of the cluster node for \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster.get_sigma_node()->get_child_pointer(j);

              /**
               * Make sure that the two clusters have the same level in their
               * respective cluster trees, i.e. level preserving property.
               */
              Assert(tau_son_node_pointer->get_level() ==
                       sigma_son_node_pointer->get_level(),
                     ExcDimensionMismatch(tau_son_node_pointer->get_level(),
                                          sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster and recursively partition from it.
               */
              BlockCluster<spacedim, Number> new_block_cluster(
                tau_son_node_pointer, sigma_son_node_pointer);
              std::vector<data_value_type> temp_block_cluster_list(
                partition_from_block_cluster(new_block_cluster,
                                             all_support_points,
                                             cell_size_at_dofs));

              /**
               * Append the resulted block cluster list obtained from the
               * recursion to the local block cluster list.
               */
              for (const auto &block_cluster : temp_block_cluster_list)
                {
                  local_block_cluster_list.push_back(block_cluster);
                }
            }
        }
    }

  return local_block_cluster_list;
}

template <int spacedim, typename Number>
std::vector<typename BlockClusterTree<spacedim, Number>::data_value_type> &
BlockClusterTree<spacedim, Number>::get_block_cluster_list()
{
  return block_cluster_list;
}

template <int spacedim, typename Number>
const std::vector<typename BlockClusterTree<spacedim, Number>::data_value_type>
  &
  BlockClusterTree<spacedim, Number>::get_block_cluster_list() const
{
  return block_cluster_list;
}

/**
 * @}
 */

#endif /* INCLUDE_BLOCK_CLUSTER_TREE_H_ */
