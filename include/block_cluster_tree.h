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
#include "debug_tools.h"
#include "lapack_full_matrix_ext.h"
#include "tree.h"

/**
 * \brief Class for block cluster tree.
 *
 * A block cluster tree is a quad-tree which holds a hierarchy of doubly linked
 * nodes with the type TreeNode. Because a node in the block cluster tree has
 * four children, the template argument \p T required by \p TreeNode should
 * be 4.
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
   * Construct from two cluster trees built from pure cardinality based
   * partition, which has no admissibility condition.
   * @param TI
   * @param TJ
   */
  BlockClusterTree(const ClusterTree<spacedim, Number> &TI,
                   const ClusterTree<spacedim, Number> &TJ);

  /**
   * Construct from two cluster trees and admissibility condition.
   *
   * The Cartesian product of the two clusters in the root nodes of \f$T(I)\f$
   * and \f$T(J)\f$ becomes the data in the root node of the block cluster tree.
   */
  BlockClusterTree(const ClusterTree<spacedim, Number> &TI,
                   const ClusterTree<spacedim, Number> &TJ,
                   Number                               eta);

  /**
   * Perform a recursive partition in tensor product form without the
   * admissibility condition because the two comprising cluster trees are built
   * from pure cardinality based partition.
   */
  void
  partition_tensor_product();

  /**
   * Perform a recursive partition in coarse non-tensor product form without the
   * admissibility condition because the two comprising cluster trees are built
   * from pure cardinality based partition.
   */
  void
  partition_coarse_non_tensor_product();

  /**
   * Perform a recursive partition in fine non-tensor product form without the
   * admissibility condition because the two comprising cluster trees are built
   * from pure cardinality based partition.
   */
  void
  partition_fine_non_tensor_product();

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
  std::vector<node_pointer_type> &
  get_leaf_set();

  /**
   * Get the reference to the block cluster list (const version).
   */
  const std::vector<node_pointer_type> &
  get_leaf_set() const;

  /**
   * Write formatted leaf set to the output stream.
   *
   * Each leaf node is written in the following format:
   *
   * >
   * [list-of-indices-in-cluster-tau],[list-of-indices-in-cluster-sigma],is_near_field
   *
   * For example,
   *
   * > [1 2 3 ...],[7 8 9 ...],1
   * @param out
   */
  void
  write_leaf_set(std::ostream &out) const;

  /**
   * Write formatted leaf set to the output stream as well as the rank of each
   * matrix block.
   *
   * Each leaf node is written in the following format:
   *
   * >
   * [list-of-indices-in-cluster-tau],[list-of-indices-in-cluster-sigma],is_near_field,rank
   *
   * For example,
   *
   * > [1 2 3 ...],[7 8 9 ...],1,1
   * @param out
   */
  template <typename Number1 = double>
  void
  write_leaf_set(std::ostream &                      out,
                 const LAPACKFullMatrixExt<Number1> &matrix,
                 const Number1 singular_value_threshold = 0.) const;

  /**
   * Get the reference to the block cluster list which belongs to the near
   * field.
   */
  std::vector<node_pointer_type> &
  get_near_field_set();

  /**
   * Get the reference to the block cluster list which belongs to the near field
   * (const version).
   */
  const std::vector<node_pointer_type> &
  get_near_field_set() const;

  /**
   * Get the reference to the block cluster list which belongs to the far
   * field.
   */
  std::vector<node_pointer_type> &
  get_far_field_set();

  /**
   * Get the reference to the block cluster list which belongs to the far field
   * (const version).
   */
  const std::vector<node_pointer_type> &
  get_far_field_set() const;

  /**
   * Get the minimum cluster size.
   */
  unsigned int
  get_n_min() const;

  /**
   * Get the tree depth.
   */
  unsigned int
  get_depth() const;

  /**
   * Get the maximum level of the tree.
   */
  int
  get_max_level() const;

  /**
   * Get the total number of clusters in the tree.
   */
  unsigned int
  get_node_num() const;

  DeclException2(
    ExcClusterLevelMismatch,
    unsigned int,
    unsigned int,
    << "The level of cluster tau " << arg1
    << " is different from that of cluster sigma" << arg2
    << " which is not allowed in a level preserving construction of a block cluster tree.");

private:
  /**
   * Perform a recursive tensor product type partition by starting from a block
   * cluster node in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type coarse partition by starting
   * from a block cluster node in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 2.2.2 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62 (2):
   * 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_coarse_non_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type fine partition of
   * \f$\mathcal{M_{\mathcal{H},k}\f$ type by starting from a block cluster node
   * in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 5 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62 (2):
   * 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_fine_non_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type fine partition of
   * \f$\mathcal{M_{\mathcal{N},k}\f$ type by starting from a block cluster node
   * in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 5 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62 (2):
   * 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_fine_non_tensor_product_from_block_cluster_node_N(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type fine partition of
   * \f$\mathcal{M_{\mathcal{N}^*,k}\f$ type by starting from a block cluster
   * node in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 5 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62 (2):
   * 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_fine_non_tensor_product_from_block_cluster_node_Nstar(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive partition by starting from a block cluster node in
   * the tree.
   *
   * N.B. The evaluation of the admissibility condition has no mesh cell size
   * correction.
   *
   * The algorithm performs an iteration over all the
   * children of the current block cluster \f$b = \tau \times \sigma\f$.
   * Because the map \f$S\f$ for generating the children of \f$b\f$ is
   * realized from a tensor product of the children of \f$\tau\f$ and
   * \f$\sigma\f$, the algorithm contains nested double loops.
   *
   * @param current_block_cluster_node The pointer to the block cluster node in
   * the tree, from which the admissible partition will be performed.
   * @param all_support_points Spatial coordinates for all the support points.
   * @param leaf_set A list of block cluster node pointers which comprise the
   * leaf set with respect to \p current_block_cluster_node.
   */
  void
  partition_from_block_cluster_node(
    node_pointer_type                   current_block_cluster_node,
    const std::vector<Point<spacedim>> &all_support_points,
    std::vector<node_pointer_type> &    leaf_set_wrt_current_node);

  /**
   * Same as above but the evaluation of the admissibility condition has mesh
   * cell size correction.
   */
  void
  partition_from_block_cluster_node(
    node_pointer_type                   current_block_cluster_node,
    const std::vector<Point<spacedim>> &all_support_points,
    const std::vector<Number> &         cell_size_at_dofs,
    std::vector<node_pointer_type> &    leaf_set_wrt_current_node);

  /**
   * Categorize the leaf set into near field set and far field set.
   */
  void
  categorize_near_and_far_field_sets();

  node_pointer_type              root_node;
  std::vector<node_pointer_type> leaf_set;
  std::vector<node_pointer_type> near_field_set;
  std::vector<node_pointer_type> far_field_set;
  unsigned int                   n_min;
  Number                         eta;

  /**
   * Depth of the tree, which is the maximum level plus one.
   */
  unsigned int depth;

  /**
   * Maximum node level in the tree, which is \p depth - 1.
   */
  int max_level;

  /**
   * Total number of block clusters in the tree.
   */
  unsigned int node_num;
};

template <int spacedim, typename Number>
std::ostream &
operator<<(std::ostream &                            out,
           const BlockClusterTree<spacedim, Number> &block_cluster_tree)
{
  out << "* Tree depth: " << block_cluster_tree.depth << "\n";
  out << "* Tree max level: " << block_cluster_tree.get_max_level() << "\n";
  out
    << "* Total number of block cluster tree nodes obtained during partition: "
    << block_cluster_tree.get_node_num() << "\n";
  out
    << "* Total number of block cluster tree nodes obtained by recursive tree traversal: "
    << CountTreeNodes(block_cluster_tree.get_root()) << "\n";
  out << "* Number of block clusters in the leaf set: "
      << block_cluster_tree.get_leaf_set().size() << "\n";

  out << "* Tree nodes:\n";
  PrintTree(out, block_cluster_tree.root_node);

  out << "* Leaf set:\n";
  print_vector_of_tree_node_pointer_values(out,
                                           block_cluster_tree.get_leaf_set(),
                                           "\n");

  out << "* Near field set: " << block_cluster_tree.get_near_field_set().size()
      << " block clusters\n";
  print_vector_of_tree_node_pointer_values(
    out, block_cluster_tree.get_near_field_set(), "\n");

  out << "* Far field set: " << block_cluster_tree.get_far_field_set().size()
      << " block clusters\n";
  print_vector_of_tree_node_pointer_values(
    out, block_cluster_tree.get_far_field_set(), "\n");

  return out;
}

template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree()
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(2)
  , eta(1.0)
  , depth(0)
  , max_level(-1)
  , node_num(0)
{}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  const ClusterTree<spacedim, Number> &TI,
  const ClusterTree<spacedim, Number> &TJ)
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(std::min(TI.get_n_min(), TJ.get_n_min()))
  , eta(1.0)
  , depth(0)
  , max_level(-1)
  , node_num(0)
{
  // Initialize the four null child pointers.
  const std::array<node_pointer_type, child_num> empty_child_pointers{nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr};
  root_node =
    CreateTreeNode<data_value_type, child_num>(data_value_type(TI.get_root(),
                                                               TJ.get_root()),
                                               0,
                                               empty_child_pointers,
                                               nullptr);

  depth     = 1;
  max_level = 0;
  node_num  = 1;

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  const ClusterTree<spacedim, Number> &TI,
  const ClusterTree<spacedim, Number> &TJ,
  Number                               eta)
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(std::min(TI.get_n_min(), TJ.get_n_min()))
  , eta(eta)
  , depth(0)
  , max_level(-1)
  , node_num(0)
{
  // Initialize the four null child pointers.
  const std::array<node_pointer_type, child_num> empty_child_pointers{nullptr,
                                                                      nullptr,
                                                                      nullptr,
                                                                      nullptr};
  root_node =
    CreateTreeNode<data_value_type, child_num>(data_value_type(TI.get_root(),
                                                               TJ.get_root()),
                                               0,
                                               empty_child_pointers,
                                               nullptr);

  depth     = 1;
  max_level = 0;
  node_num  = 1;

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}

template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::~BlockClusterTree()
{
  DeleteTree(root_node);
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_tensor_product()
{
  partition_tensor_product_from_block_cluster_node(root_node, leaf_set);

  categorize_near_and_far_field_sets();

  depth     = calc_depth(root_node);
  max_level = depth - 1;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_coarse_non_tensor_product()
{
  partition_coarse_non_tensor_product_from_block_cluster_node(root_node,
                                                              leaf_set);

  categorize_near_and_far_field_sets();

  depth     = calc_depth(root_node);
  max_level = depth - 1;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_fine_non_tensor_product()
{
  partition_fine_non_tensor_product_from_block_cluster_node(root_node,
                                                            leaf_set);

  categorize_near_and_far_field_sets();

  depth     = calc_depth(root_node);
  max_level = depth - 1;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<Point<spacedim>> &all_support_points)
{
  partition_from_block_cluster_node(root_node, all_support_points, leaf_set);

  categorize_near_and_far_field_sets();

  depth     = calc_depth(root_node);
  max_level = depth - 1;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number> &         cell_size_at_dofs)
{
  partition_from_block_cluster_node(root_node,
                                    all_support_points,
                                    cell_size_at_dofs,
                                    leaf_set);

  categorize_near_and_far_field_sets();

  depth     = calc_depth(root_node);
  max_level = depth - 1;
}


template <int spacedim, typename Number>
typename BlockClusterTree<spacedim, Number>::node_pointer_type
BlockClusterTree<spacedim, Number>::get_root() const
{
  return root_node;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::
  partition_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_small(n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child and recursively
               * partition from it.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_tensor_product_from_block_cluster_node(
                child_block_cluster_node, leaf_set_wrt_child_node);

              /**
               * Merge the leaf set wrt. the child block cluster node into the
               * leaf set of the current block cluster node.
               */
              for (node_pointer_type block_cluster_node :
                   leaf_set_wrt_child_node)
                {
                  leaf_set_wrt_current_node.push_back(block_cluster_node);
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::
  partition_coarse_non_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_small(n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child. Then append this new
               * node as one of the children of the current block cluster node.
               * Finally, recursively partition from this node if the two
               * component clusters have the same indices, i.e. \f$I_1 \times
               * I_1\f$ and \f$I_2 \times I_2\f$; otherwise, for \f$I_1 \times
               * I_2\f$ and \f$I_2 \times I_1\f$, stop the recursion and
               * directly add them to the leaf set.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == j)
                {
                  /**
                   * Handle the case for \f$I_1 \times I_1\f$ and \f$I_2 \times
                   * I_2\f$.
                   */
                  std::vector<node_pointer_type> leaf_set_wrt_child_node;
                  partition_coarse_non_tensor_product_from_block_cluster_node(
                    child_block_cluster_node, leaf_set_wrt_child_node);

                  /**
                   * Merge the leaf set wrt. the child block cluster node into
                   * the leaf set of the current block cluster node.
                   */
                  for (node_pointer_type block_cluster_node :
                       leaf_set_wrt_child_node)
                    {
                      leaf_set_wrt_current_node.push_back(block_cluster_node);
                    }
                }
              else
                {
                  /**
                   * Handle the case for \f$I_1 \times I_2\f$ and \f$I_2 \times
                   * I_1\f$. Because the recursion stops here, we need to check
                   * and update its near field property.
                   */
                  child_block_cluster_node->get_data_pointer()->is_small(n_min);
                  leaf_set_wrt_current_node.push_back(child_block_cluster_node);
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::
  partition_fine_non_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_small(n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child. Then append this new
               * node as one of the children of the current block cluster node.
               * Finally, recursively partition from this node if the two
               * component clusters have the same indices, i.e. \f$I_1 \times
               * I_1\f$ and \f$I_2 \times I_2\f$; otherwise, for \f$I_1 \times
               * I_2\f$ and \f$I_2 \times I_1\f$, stop the recursion and
               * directly add them to the leaf set.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == j)
                {
                  /**
                   * Handle the case for \f$I_1 \times I_1\f$ and \f$I_2 \times
                   * I_2\f$.
                   */
                  std::vector<node_pointer_type> leaf_set_wrt_child_node;
                  partition_fine_non_tensor_product_from_block_cluster_node(
                    child_block_cluster_node, leaf_set_wrt_child_node);

                  /**
                   * Merge the leaf set wrt. the child block cluster node into
                   * the leaf set of the current block cluster node.
                   */
                  for (node_pointer_type block_cluster_node :
                       leaf_set_wrt_child_node)
                    {
                      leaf_set_wrt_current_node.push_back(block_cluster_node);
                    }
                }
              else
                {
                  if (i == 0)
                    {
                      /**
                       * Handle the case for \f$I_1 \times I_2\f$ and perform
                       * the \f$\mathcal{N}\f$-type partition.
                       */
                      std::vector<node_pointer_type> leaf_set_wrt_child_node;
                      partition_fine_non_tensor_product_from_block_cluster_node_N(
                        child_block_cluster_node, leaf_set_wrt_child_node);

                      /**
                       * Merge the leaf set wrt. the child block cluster node
                       * into the leaf set of the current block cluster node.
                       */
                      for (node_pointer_type block_cluster_node :
                           leaf_set_wrt_child_node)
                        {
                          leaf_set_wrt_current_node.push_back(
                            block_cluster_node);
                        }
                    }
                  else if (i == 1)
                    {
                      /**
                       * Handle the case for \f$I_2 \times I_1\f$ and perform
                       * the \f$\mathcal{N}^*\f$-type partition.
                       */
                      std::vector<node_pointer_type> leaf_set_wrt_child_node;
                      partition_fine_non_tensor_product_from_block_cluster_node_Nstar(
                        child_block_cluster_node, leaf_set_wrt_child_node);

                      /**
                       * Merge the leaf set wrt. the child block cluster node
                       * into the leaf set of the current block cluster node.
                       */
                      for (node_pointer_type block_cluster_node :
                           leaf_set_wrt_child_node)
                        {
                          leaf_set_wrt_current_node.push_back(
                            block_cluster_node);
                        }
                    }
                  else
                    {
                      Assert(false, ExcInternalError());
                    }
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::
  partition_fine_non_tensor_product_from_block_cluster_node_N(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_small(n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child. Then append this new
               * node as one of the children of the current block cluster node.
               * Finally, recursively partition from this node if the two
               * component clusters have the same indices, i.e. \f$I_1 \times
               * I_1\f$ and \f$I_2 \times I_2\f$; otherwise, for \f$I_1 \times
               * I_2\f$ and \f$I_2 \times I_1\f$, stop the recursion and
               * directly add them to the leaf set.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == 1 && j == 0)
                {
                  /**
                   * Handle the case for \f$I_2 \times I_1\f$ and perform the
                   * \f$\mathcal{N}\f$-type partition.
                   */
                  std::vector<node_pointer_type> leaf_set_wrt_child_node;
                  partition_fine_non_tensor_product_from_block_cluster_node_N(
                    child_block_cluster_node, leaf_set_wrt_child_node);

                  /**
                   * Merge the leaf set wrt. the child block cluster node into
                   * the leaf set of the current block cluster node.
                   */
                  for (node_pointer_type block_cluster_node :
                       leaf_set_wrt_child_node)
                    {
                      leaf_set_wrt_current_node.push_back(block_cluster_node);
                    }
                }
              else
                {
                  /**
                   * Handle the case for \f$I_1 \times I_1\f$, \f$I_1 \times
                   * I_2\f$ and \f$I_2 \times I_2\f$. Because the recursion
                   * stops here, we need to check and update its near field
                   * property.
                   */
                  child_block_cluster_node->get_data_pointer()->is_small(n_min);
                  leaf_set_wrt_current_node.push_back(child_block_cluster_node);
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::
  partition_fine_non_tensor_product_from_block_cluster_node_Nstar(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_small(n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child. Then append this new
               * node as one of the children of the current block cluster node.
               * Finally, recursively partition from this node if the two
               * component clusters have the same indices, i.e. \f$I_1 \times
               * I_1\f$ and \f$I_2 \times I_2\f$; otherwise, for \f$I_1 \times
               * I_2\f$ and \f$I_2 \times I_1\f$, stop the recursion and
               * directly add them to the leaf set.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == 0 && j == 1)
                {
                  /**
                   * Handle the case for \f$I_1 \times I_2\f$ and perform the
                   * \f$\mathcal{N}^*\f$-type partition.
                   */
                  std::vector<node_pointer_type> leaf_set_wrt_child_node;
                  partition_fine_non_tensor_product_from_block_cluster_node_Nstar(
                    child_block_cluster_node, leaf_set_wrt_child_node);

                  /**
                   * Merge the leaf set wrt. the child block cluster node into
                   * the leaf set of the current block cluster node.
                   */
                  for (node_pointer_type block_cluster_node :
                       leaf_set_wrt_child_node)
                    {
                      leaf_set_wrt_current_node.push_back(block_cluster_node);
                    }
                }
              else
                {
                  /**
                   * Handle the case for \f$I_1 \times I_1\f$, \f$I_2 \times
                   * I_1\f$ and \f$I_2 \times I_2\f$. Because the recursion
                   * stops here, we need to check and update its near field
                   * property.
                   */
                  child_block_cluster_node->get_data_pointer()->is_small(n_min);
                  leaf_set_wrt_current_node.push_back(child_block_cluster_node);
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_from_block_cluster_node(
  node_pointer_type                   current_block_cluster_node,
  const std::vector<Point<spacedim>> &all_support_points,
  std::vector<node_pointer_type> &    leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_admissible_or_small(
        eta, all_support_points, n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child and recursively
               * partition from it.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_from_block_cluster_node(child_block_cluster_node,
                                                all_support_points,
                                                leaf_set_wrt_child_node);

              /**
               * Merge the leaf set wrt. the child block cluster node into the
               * leaf set of the current block cluster node.
               */
              for (node_pointer_type block_cluster_node :
                   leaf_set_wrt_child_node)
                {
                  leaf_set_wrt_current_node.push_back(block_cluster_node);
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_from_block_cluster_node(
  node_pointer_type                   current_block_cluster_node,
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number> &         cell_size_at_dofs,
  std::vector<node_pointer_type> &    leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_admissible_or_small(
        eta, all_support_points, cell_size_at_dofs, n_min))
    {
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
    }
  else
    {
      unsigned int child_counter = 0;

      // Iterate over each child of the cluster \f$\tau\f$.
      for (unsigned int i = 0; i < (ClusterTree<spacedim, Number>::child_num);
           i++)
        {
          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_son_node_pointer =
              current_block_cluster_node->get_data_pointer()
                ->get_tau_node()
                ->get_child_pointer(i);

          // Iterate over each child of the cluster \f$\sigma\f$.
          for (unsigned int j = 0;
               j < (ClusterTree<spacedim, Number>::child_num);
               j++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer =
                  current_block_cluster_node->get_data_pointer()
                    ->get_sigma_node()
                    ->get_child_pointer(j);

              /**
               * Make sure that the two clusters \f$\tau\f$ and \f$\sigma\f$
               * have the same level in their respective cluster trees, i.e.
               * level preserving property should be satisfied.
               */
              Assert(
                tau_son_node_pointer->get_level() ==
                  sigma_son_node_pointer->get_level(),
                ExcClusterLevelMismatch(tau_son_node_pointer->get_level(),
                                        sigma_son_node_pointer->get_level()));

              /**
               * Create a new block cluster node as child and recursively
               * partition from it.
               */
              const std::array<node_pointer_type, child_num>
                                empty_child_pointers{nullptr, nullptr, nullptr, nullptr};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type, child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_from_block_cluster_node(child_block_cluster_node,
                                                all_support_points,
                                                cell_size_at_dofs,
                                                leaf_set_wrt_child_node);

              /**
               * Merge the leaf set wrt. the child block cluster node into the
               * leaf set of the current block cluster node.
               */
              for (node_pointer_type block_cluster_node :
                   leaf_set_wrt_child_node)
                {
                  leaf_set_wrt_current_node.push_back(block_cluster_node);
                }
            }
        }
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::categorize_near_and_far_field_sets()
{
  near_field_set.clear();
  far_field_set.clear();

  for (node_pointer_type node : leaf_set)
    {
      if (node->get_data_reference().get_is_near_field())
        {
          near_field_set.push_back(node);
        }
      else
        {
          far_field_set.push_back(node);
        }
    }
}


template <int spacedim, typename Number>
std::vector<typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_leaf_set()
{
  return leaf_set;
}


template <int spacedim, typename Number>
const std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_leaf_set() const
{
  return leaf_set;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::write_leaf_set(std::ostream &out) const
{
  for (node_pointer_type bc_node : leaf_set)
    {
      /**
       * Print index set of cluster \f$\tau\f$.
       */
      out << "[";
      print_vector_values(out,
                          bc_node->get_data_reference()
                            .get_tau_node()
                            ->get_data_reference()
                            .get_index_set(),
                          " ",
                          false);
      out << "],";

      /**
       * Print index set of cluster \f$\sigma\f$.
       */
      out << "[";
      print_vector_values(out,
                          bc_node->get_data_reference()
                            .get_sigma_node()
                            ->get_data_reference()
                            .get_index_set(),
                          " ",
                          false);
      out << "],";

      /**
       * Print the \p is_near_field flag.
       */
      out << (bc_node->get_data_reference().get_is_near_field() ? 1 : 0)
          << "\n";
    }
}


template <int spacedim, typename Number>
template <typename Number1>
void
BlockClusterTree<spacedim, Number>::write_leaf_set(
  std::ostream &                      out,
  const LAPACKFullMatrixExt<Number1> &matrix,
  const Number1                       singular_value_threshold) const
{
  for (node_pointer_type bc_node : leaf_set)
    {
      const std::vector<types::global_dof_index> &tau_index_set =
        bc_node->get_data_reference()
          .get_tau_node()
          ->get_data_reference()
          .get_index_set();
      const std::vector<types::global_dof_index> &sigma_index_set =
        bc_node->get_data_reference()
          .get_sigma_node()
          ->get_data_reference()
          .get_index_set();

      /**
       * Print index set of cluster \f$\tau\f$.
       */
      out << "[";
      print_vector_values(out, tau_index_set, " ", false);
      out << "],";

      /**
       * Print index set of cluster \f$\sigma\f$.
       */
      out << "[";
      print_vector_values(out, sigma_index_set, " ", false);
      out << "],";

      /**
       * Print the \p is_near_field flag.
       */
      out << (bc_node->get_data_reference().get_is_near_field() ? 1 : 0) << ",";

      /**
       * Make a local copy of the matrix block and calculate its rank using SVD.
       */
      const size_t                 nrows = tau_index_set.size();
      const size_t                 ncols = sigma_index_set.size();
      LAPACKFullMatrixExt<Number1> local_matrix(nrows, ncols);

      for (size_t i = 0; i < nrows; i++)
        {
          for (size_t j = 0; j < ncols; j++)
            {
              local_matrix(i, j) = matrix(tau_index_set[i], sigma_index_set[j]);
            }
        }

      const size_t rank = local_matrix.rank(singular_value_threshold);

      /**
       * Print the \p rank flag.
       */
      out << rank << "\n";
    }
}


template <int spacedim, typename Number>
std::vector<typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_near_field_set()
{
  return near_field_set;
}


template <int spacedim, typename Number>
const std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_near_field_set() const
{
  return near_field_set;
}


template <int spacedim, typename Number>
std::vector<typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_far_field_set()
{
  return far_field_set;
}


template <int spacedim, typename Number>
const std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_far_field_set() const
{
  return far_field_set;
}


template <int spacedim, typename Number>
unsigned int
BlockClusterTree<spacedim, Number>::get_depth() const
{
  return depth;
}


template <int spacedim, typename Number>
int
BlockClusterTree<spacedim, Number>::get_max_level() const
{
  return max_level;
}


template <int spacedim, typename Number>
unsigned int
BlockClusterTree<spacedim, Number>::get_node_num() const
{
  return node_num;
}


template <int spacedim, typename Number>
unsigned int
BlockClusterTree<spacedim, Number>::get_n_min() const
{
  return n_min;
}

/**
 * @}
 */

#endif /* INCLUDE_BLOCK_CLUSTER_TREE_H_ */
