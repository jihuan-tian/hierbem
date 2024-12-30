/**
 * \file block_cluster_tree.h
 * \brief Implementation of the class BlockClusterTree.
 * \ingroup hierarchical_matrices
 * \date 2021-04-20
 * \author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_BLOCK_CLUSTER_TREE_H_
#define HIERBEM_INCLUDE_BLOCK_CLUSTER_TREE_H_

/**
 * \ingroup hierarchical_matrices
 * @{
 */

#include <deal.II/base/memory_consumption.h>

#include "block_cluster.h"
#include "cluster_tree.h"
#include "config.h"
#include "debug_tools.h"
#include "lapack_full_matrix_ext.h"
#include "tree.h"

HBEM_NS_OPEN

/**
 * \brief Class for block cluster tree.
 *
 * A block cluster tree is a quad-tree which holds a hierarchy of doubly
 * linked nodes with the type TreeNode. Because a node in the block cluster
 * tree has
 * four children, the template argument \p T required by \p TreeNode should
 * be 4.
 */
template <int spacedim, typename Number = double>
class BlockClusterTree
{
public:
  /**
   * Number of children in a block cluster tree.
   *
   * At present, only quad-tree is allowed.
   */
  inline static const unsigned int child_num = 4;

  /**
   * Print a whole block cluster tree using recursion.
   * @param out
   * @param block_cluster_tree
   * @return
   */
  template <int spacedim1, typename Number1>
  friend std::ostream &
  operator<<(std::ostream                               &out,
             const BlockClusterTree<spacedim1, Number1> &block_cluster_tree);

  template <int spacedim1, typename Number1>
  friend void
  split_block_cluster_node(
    TreeNode<BlockCluster<spacedim1, Number1>,
             BlockClusterTree<spacedim1, Number1>::child_num> *bc_node,
    BlockClusterTree<spacedim1, Number1>                      &bc_tree,
    const TreeNodeSplitMode                                    split_mode,
    const bool if_add_child_nodes_to_leaf_set);

  template <int spacedim1, typename Number1>
  friend bool
  prune_to_partition_recursion(
    BlockClusterTree<spacedim1, Number1> &bct,
    const std::vector<
      TreeNode<BlockCluster<spacedim1, Number1>,
               BlockClusterTree<spacedim1, Number1>::child_num> *> &partition,
    TreeNode<BlockCluster<spacedim1, Number1>,
             BlockClusterTree<spacedim1, Number1>::child_num>      *bc_node);

  /**
   * Data type of the tree node.
   */
  typedef TreeNode<BlockCluster<spacedim, Number>,
                   BlockClusterTree<spacedim, Number>::child_num>
    node_value_type;
  typedef TreeNode<BlockCluster<spacedim, Number>,
                   BlockClusterTree<spacedim, Number>::child_num>
    *node_pointer_type;
  typedef const TreeNode<BlockCluster<spacedim, Number>,
                         BlockClusterTree<spacedim, Number>::child_num>
    *node_const_pointer_type;
  typedef TreeNode<BlockCluster<spacedim, Number>,
                   BlockClusterTree<spacedim, Number>::child_num>
    &node_reference_type;
  typedef const TreeNode<BlockCluster<spacedim, Number>,
                         BlockClusterTree<spacedim, Number>::child_num>
    &node_const_reference_type;

  /**
   * Data type of the data held by a tree node.
   */
  typedef BlockCluster<spacedim, Number>        data_value_type;
  typedef BlockCluster<spacedim, Number>       *data_pointer_type;
  typedef const BlockCluster<spacedim, Number> *data_const_pointer_type;
  typedef BlockCluster<spacedim, Number>       &data_reference_type;
  typedef const BlockCluster<spacedim, Number> &data_const_reference_type;

  template <int spacedim1, typename Number1>
  friend TreeNode<BlockCluster<spacedim1, Number1>,
                  BlockClusterTree<spacedim1, Number1>::child_num> *
  find_bc_node_in_partition_intersect_current_bc_node(
    TreeNode<BlockCluster<spacedim1, Number1>,
             BlockClusterTree<spacedim1, Number1>::child_num> *current_bc_node,
    const std::vector<
      TreeNode<BlockCluster<spacedim1, Number1>,
               BlockClusterTree<spacedim1, Number1>::child_num> *> &partition);

  template <int spacedim1, typename Number1>
  friend TreeNode<BlockCluster<spacedim1, Number1>,
                  BlockClusterTree<spacedim1, Number1>::child_num> *
  find_bc_node_in_partition_proper_subset_of_current_bc_node(
    TreeNode<BlockCluster<spacedim1, Number1>,
             BlockClusterTree<spacedim1, Number1>::child_num> *current_bc_node,
    const std::vector<
      TreeNode<BlockCluster<spacedim1, Number1>,
               BlockClusterTree<spacedim1, Number1>::child_num> *> &partition);

  template <int spacedim1, typename Number1>
  friend void
  print_block_cluster_node_info_as_dot_node(
    std::ostream                                                    &out,
    const TreeNode<BlockCluster<spacedim1, Number1>,
                   BlockClusterTree<spacedim1, Number1>::child_num> &tree_node);

  /**
   * Default constructor, which initializes an empty quad-tree.
   */
  BlockClusterTree();

  /**
   * Construct from two cluster trees built from pure cardinality based
   * partition, which has no admissibility condition.
   *
   * After construction, there is only one node in the block cluster tree,
   * i.e. tree hierarchy is not built.
   *
   * @param TI
   * @param TJ
   * @param n_min
   */
  BlockClusterTree(const ClusterTree<spacedim, Number> &TI,
                   const ClusterTree<spacedim, Number> &TJ,
                   const unsigned int                   n_min = 0);

  /**
   * Construct from two cluster nodes, whose Cartesian product is the root
   * node of the block cluster tree.
   *
   * After construction, there is only one node in the block cluster tree,
   * i.e. tree hierarchy is not built.
   *
   * @param tau_root_node
   * @param sigma_root_node
   */
  BlockClusterTree(
    typename ClusterTree<spacedim, Number>::node_pointer_type tau_root_node,
    typename ClusterTree<spacedim, Number>::node_pointer_type sigma_root_node,
    const unsigned int                                        n_min = 1);

  /**
   * Construct a block cluster subtree by initializing from a block cluster
   * node.
   *
   * The constructed block cluster subtree simply takes over an existing block
   * cluster node in a block cluster tree which has been built before.
   *
   * @param bc_node
   * @param n_min
   */
  BlockClusterTree(node_pointer_type bc_node, const unsigned int n_min);

  /**
   * Construct a block cluster subtree by initializing from a block cluster
   * node. This version has the admissibility condition.
   *
   * The constructed block cluster subtree simply takes over an existing block
   * cluster node in a block cluster tree which has been built before.
   *
   * @param bc_node
   */
  BlockClusterTree(node_pointer_type  bc_node,
                   const Number       eta,
                   const unsigned int n_min);

  /**
   * Construct from two cluster trees and admissibility condition.
   *
   * The Cartesian product of the two clusters in the root nodes of \f$T(I)\f$
   * and \f$T(J)\f$ becomes the data in the root node of the block cluster
   * tree.
   *
   * After construction, there is only one node in the block cluster tree,
   * i.e. tree hierarchy is not built.
   */
  BlockClusterTree(const ClusterTree<spacedim, Number> &TI,
                   const ClusterTree<spacedim, Number> &TJ,
                   const Number                         eta,
                   const unsigned int                   n_min);

  /**
   * Copy constructor for \p BlockClusterTree, which realizes deep copy
   * internally.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>Because this is deep copy, the constructed block cluster tree is a
   * new tree but never a subtree of an existing block cluster tree. Hence,
   * the
   * data field \p is_subtree is set to \p false.</dd>
   * </dl>
   *
   * @param bct
   */
  BlockClusterTree(const BlockClusterTree<spacedim, Number> &bct);

  /**
   * Copy constructor via shallow copy.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The data members of the input \p bct must be cleared before exiting
   * this constructor. Otherwise, when \p bct is out of its range, the data
   * associated with the current block cluster tree will also be destroyed,
   * which is undesired.</dd>
   * </dl>
   *
   * @param bct
   * @return
   */
  BlockClusterTree(BlockClusterTree<spacedim, Number> &&bct);

  /**
   * Overloaded assignment operator with deep copy.
   * @param bct
   * @return
   */
  BlockClusterTree<spacedim, Number> &
  operator=(const BlockClusterTree<spacedim, Number> &bct);

  /**
   * Overloaded assignment operator with shallow copy.
   * @param bct
   * @return
   */
  BlockClusterTree<spacedim, Number> &
  operator=(BlockClusterTree<spacedim, Number> &&bct);

  /**
   * Destructor which recursively destroys every node in the block cluster
   * tree.
   */
  ~BlockClusterTree();

  /**
   * Release the memory of the block cluster tree.
   */
  void
  release();

  /**
   * Clear the data field of the block cluster tree because its memory has
   * been migrated to another object via shallow copy.
   */
  void
  clear();

  /**
   * Perform a recursive partition in tensor product form without the
   * admissibility condition because the two comprising cluster trees are
   * built from pure cardinality based partition.
   */
  void
  partition_tensor_product();

  /**
   * Perform a recursive partition in coarse non-tensor product form without
   * the admissibility condition because the two comprising cluster trees are
   * built from pure cardinality based partition.
   */
  void
  partition_coarse_non_tensor_product();

  /**
   * Perform a recursive partition in fine non-tensor product form without the
   * admissibility condition because the two comprising cluster trees are
   * built from pure cardinality based partition.
   */
  void
  partition_fine_non_tensor_product();

  /**
   * Perform a recursive partition by starting from the root node. The
   * evaluation of the admissibility condition has no mesh cell size
   * correction.
   *
   * \mynote{The index sets held by the two clusters share a same external
   * DoF numbering.}
   *
   * @param all_support_points All the support points.
   */
  void
  partition(const std::vector<types::global_dof_index>
              &internal_to_external_dof_numbering,
            const std::vector<Point<spacedim>> &all_support_points);

  /**
   * Perform a recursive partition by starting from the root node. The
   * evaluation of the admissibility condition has no mesh cell size
   * correction.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters share a same internal DoF numbering, which need to be mapped to
   * the external numbering for accessing the list of support point
   * coordinates.}
   *
   * @param internal_to_external_dof_numbering_I
   * @param internal_to_external_dof_numbering_J
   * @param all_support_points_in_I
   * @param all_support_points_in_J
   */
  void
  partition(const std::vector<types::global_dof_index>
              &internal_to_external_dof_numbering_I,
            const std::vector<types::global_dof_index>
              &internal_to_external_dof_numbering_J,
            const std::vector<Point<spacedim>> &all_support_points_in_I,
            const std::vector<Point<spacedim>> &all_support_points_in_J);

  /**
   * Perform a recursive partition by starting from the root node. The
   * evaluation of the admissibility condition has mesh cell size correction.
   *
   * \mynote{The index sets held by the two clusters share a same external
   * DoF numbering.}
   *
   * @param all_support_points All the support points.
   */
  void
  partition(const std::vector<types::global_dof_index>
              &internal_to_external_dof_numbering,
            const std::vector<Point<spacedim>> &all_support_points,
            const std::vector<Number>          &cell_size_at_dofs);

  /**
   * Perform a recursive partition by starting from the root node. The
   * evaluation of the admissibility condition has mesh cell size correction.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters share a same internal DoF numbering, which need to be mapped to
   * the external numbering for accessing the list of support point
   * coordinates.}
   *
   * @param internal_to_external_dof_numbering_I
   * @param internal_to_external_dof_numbering_J
   * @param all_support_points_in_I
   * @param all_support_points_in_J
   * @param cell_size_at_dofs_in_I
   * @param cell_size_at_dofs_in_J
   */
  void
  partition(const std::vector<types::global_dof_index>
              &internal_to_external_dof_numbering_I,
            const std::vector<types::global_dof_index>
              &internal_to_external_dof_numbering_J,
            const std::vector<Point<spacedim>> &all_support_points_in_I,
            const std::vector<Point<spacedim>> &all_support_points_in_J,
            const std::vector<Number>          &cell_size_at_dofs_in_I,
            const std::vector<Number>          &cell_size_at_dofs_in_J);

  /**
   * Extend the current block cluster tree to be finer than the given
   * partition.
   *
   * This member functions implements (7.10a) in Hackbusch's
   * \hmatrix book.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd><strong>This algorithm iterates over each element in the leaf set
   * of the block cluster tree to be extended. During the iteration, because
   * leaf nodes may be further split into smaller ones, the leaf set is not a
   * constant. Hence, the leaf set should be updated immediately whenever a
   * block cluster node is split. This behavior is different from other
   * functions such as \p HMatrix<spacedim, Number>::refine_to_supertree, the
   * leaf set of of which will be built after the whole tree hierarchy has
   * been constructed.</strong>
   *   </dd>
   * </dl>
   * @param P
   * @return Whether the block cluster tree has really been extended.
   */
  bool
  extend_finer_than_partition(const std::vector<node_pointer_type> &partition);

  /**
   * Extend the current block cluster tree to the given finer partition.
   *
   * This member functions implements (7.10b) in Hackbusch's
   * \hmatrix book.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd><strong>This algorithm iterates over each element in the leaf set
   * of the block cluster tree to be extended. During the iteration, because
   * leaf nodes may be further split into smaller ones, the leaf set is not a
   * constant. Hence, the leaf set should be updated immediately whenever a
   * block cluster node is split. This behavior is different from other
   * functions such as \p HMatrix<spacedim, Number>::refine_to_supertree, the
   * leaf set of of which will be built after the whole tree hierarchy has
   * been constructed.</strong>
   *   </dd>
   * </dl>
   */
  bool
  extend_to_finer_partition(const std::vector<node_pointer_type> &partition);

  /**
   * Prune the descendants of the given \bcn.
   * @param bc_node
   * @return Whether the \bct is really pruned.
   */
  bool
  prune_descendants_from_node(node_pointer_type bc_node);

  /**
   * Prune all those \bc nodes which are descendants of the \bc nodes in the
   * given partition.
   *
   * @param partition
   * @param is_partition_in_bct Whether the \bc nodes in the given \p partition
   * are contained in this \bct.
   */
  bool
  prune_to_partition(const std::vector<node_pointer_type> &partition,
                     const bool                            is_partition_in_bct);

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
   * Build the leaf set by tree recursion.
   */
  void
  build_leaf_set();

  /**
   * Write formatted leaf set to the output stream.
   *
   * Each leaf node is written in the following format:
   *
   * >
   * [index-range-in-cluster-tau],[index-range-in-cluster-sigma],is_near_field
   *
   * For example,
   *
   * > [0 15),[15 20),1
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
   * [index-range-in-cluster-tau],[index-range-in-cluster-sigma],is_near_field,rank
   *
   * For example,
   *
   * > [0 15),[15 20),1,1
   * @param out
   */
  template <typename Number1 = double>
  void
  write_leaf_set(std::ostream                       &out,
                 const LAPACKFullMatrixExt<Number1> &matrix,
                 const Number1 singular_value_threshold = 0.) const;

  /**
   * Print the \bct hierarchy information as directional graph in Graphviz
   * dot format.
   *
   * @param out
   */
  void
  print_bct_info_as_dot(std::ostream &out) const;

  /**
   * Get the reference to the block cluster list which belongs to the near
   * field.
   */
  std::vector<node_pointer_type> &
  get_near_field_set();

  /**
   * Get the reference to the block cluster list which belongs to the near
   * field (const version).
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
   * Get the reference to the block cluster list which belongs to the far
   * field (const version).
   */
  const std::vector<node_pointer_type> &
  get_far_field_set() const;

  /**
   * Get the minimum cluster size.
   */
  unsigned int
  get_n_min() const;

  /**
   * Get the admissibility parameter.
   * @return
   */
  Number
  get_eta() const;

  /**
   * Get the tree depth.
   */
  unsigned int
  get_depth() const;

  /**
   * Calculate the depth.
   */
  void
  calc_depth_and_max_level();

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

  /**
   * Set the total number of nodes in the tree.
   */
  void
  set_node_num(unsigned int node_num);

  /**
   * Increase the total number of nodes in the tree.
   */
  unsigned int
  increase_node_num(unsigned int increased_node_num = 1);

  /**
   * Decrease the total number of nodes in the tree.
   */
  unsigned int
  decrease_node_num(unsigned int decreased_node_num = 1);

  DeclException2(
    ExcClusterLevelMismatch,
    unsigned int,
    unsigned int,
    << "The level of cluster tau " << arg1
    << " is different from that of cluster sigma" << arg2
    << " which is not allowed in a level preserving construction of a block cluster tree.");

  /**
   * Categorize the leaf set into near field set and far field set.
   */
  void
  categorize_near_and_far_field_sets();

  /**
   * Estimate the memory consumption of the block cluster tree with all nodes
   * and contained data.
   */
  std::size_t
  memory_consumption() const;

  /**
   * Estimate the memory consumption of all nodes in the cluster tree and
   * their contained data.
   */
  std::size_t
  memory_consumption_of_all_block_clusters() const;

private:
  /**
   * Perform a recursive tensor product type partition by starting from a
   * block cluster node in the tree.
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
   * Reference: Section 2.2.2 in Hackbusch, W. 1999. “A Sparse Matrix
   * Arithmetic Based on H-Matrices. Part I: Introduction to H-Matrices.”
   * Computing 62 (2): 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_coarse_non_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type fine partition of
   * \f$\mathcal{M}_{\mathcal{H},k}\f$ type by starting from a block cluster
   * node in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 5 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62
   * (2): 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_fine_non_tensor_product_from_block_cluster_node(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type fine partition of
   * \f$\mathcal{M}_{\mathcal{N},k}\f$ type by starting from a block cluster
   * node in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 5 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62
   * (2): 89–108.
   * @param current_block_cluster_node
   * @param leaf_set_wrt_current_node
   */
  void
  partition_fine_non_tensor_product_from_block_cluster_node_N(
    node_pointer_type               current_block_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node);

  /**
   * Perform a recursive non-tensor product type fine partition of
   * \f$\mathcal{M}_{\mathcal{N}^*,k}\f$ type by starting from a block cluster
   * node in the tree.
   *
   * No admissibility condition is enabled in this situation, because the two
   * comprising cluster trees are built from pure cardinality based partition.
   *
   * Reference: Section 5 in Hackbusch, W. 1999. “A Sparse Matrix Arithmetic
   * Based on H-Matrices. Part I: Introduction to H-Matrices.” Computing 62
   * (2): 89–108.
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
   * \mynote{The index sets held by the two clusters share a same external
   * DoF numbering.}
   *
   * The algorithm performs an iteration over all the
   * children of the current block cluster \f$b = \tau \times \sigma\f$.
   * Because the map \f$S\f$ for generating the children of \f$b\f$ is
   * realized from a tensor product of the children of \f$\tau\f$ and
   * \f$\sigma\f$, the algorithm contains nested double loops.
   *
   * @param current_block_cluster_node The pointer to the block cluster node in
   * the tree, from which the admissible partition will be performed.
   * @param internal_to_external_dof_numbering
   * @param all_support_points Spatial coordinates for all the support points.
   * @param leaf_set A list of block cluster node pointers which comprise the
   * leaf set with respect to \p current_block_cluster_node.
   */
  void
  partition_from_block_cluster_node(
    node_pointer_type current_block_cluster_node,
    const std::vector<types::global_dof_index>
                                       &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim>> &all_support_points,
    std::vector<node_pointer_type>     &leaf_set_wrt_current_node);

  /**
   * Perform a recursive partition by starting from a block cluster node in
   * the tree.
   *
   * N.B. The evaluation of the admissibility condition has no mesh cell size
   * correction.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters share a same internal DoF numbering, which need to be mapped to
   * the external numbering for accessing the list of support point
   * coordinates.}
   *
   * @param current_block_cluster_node
   * @param internal_to_external_dof_numbering_I
   * @param internal_to_external_dof_numbering_J
   * @param all_support_points_in_I
   * @param all_support_points_in_J
   * @param leaf_set_wrt_current_node
   */
  void
  partition_from_block_cluster_node(
    node_pointer_type current_block_cluster_node,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_I,
    const std::vector<types::global_dof_index>
                                       &internal_to_external_dof_numbering_J,
    const std::vector<Point<spacedim>> &all_support_points_in_I,
    const std::vector<Point<spacedim>> &all_support_points_in_J,
    std::vector<node_pointer_type>     &leaf_set_wrt_current_node);

  /**
   * Perform a recursive partition by starting from a block cluster node in
   * the tree.
   *
   * N.B. The evaluation of the admissibility condition has the mesh cell size
   * correction.
   *
   * \mynote{The index sets held by the two clusters share a same external
   * DoF numbering.}
   *
   * @param current_block_cluster_node
   * @param internal_to_external_dof_numbering
   * @param all_support_points
   * @param cell_size_at_dofs
   * @param leaf_set_wrt_current_node
   */
  void
  partition_from_block_cluster_node(
    node_pointer_type current_block_cluster_node,
    const std::vector<types::global_dof_index>
                                       &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim>> &all_support_points,
    const std::vector<Number>          &cell_size_at_dofs,
    std::vector<node_pointer_type>     &leaf_set_wrt_current_node);

  /**
   * Perform a recursive partition by starting from a block cluster node in
   * the tree.
   *
   * N.B. The evaluation of the admissibility condition has the mesh cell size
   * correction.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters share a same internal DoF numbering, which need to be mapped to
   * the external numbering for accessing the list of support point
   * coordinates.}
   *
   * @param current_block_cluster_node
   * @param internal_to_external_dof_numbering_I
   * @param internal_to_external_dof_numbering_J
   * @param all_support_points_in_I
   * @param all_support_points_in_J
   * @param cell_size_at_dofs_in_I
   * @param cell_size_at_dofs_in_J
   * @param leaf_set_wrt_current_node
   */
  void
  partition_from_block_cluster_node(
    node_pointer_type current_block_cluster_node,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_I,
    const std::vector<types::global_dof_index>
                                       &internal_to_external_dof_numbering_J,
    const std::vector<Point<spacedim>> &all_support_points_in_I,
    const std::vector<Point<spacedim>> &all_support_points_in_J,
    const std::vector<Number>          &cell_size_at_dofs_in_I,
    const std::vector<Number>          &cell_size_at_dofs_in_J,
    std::vector<node_pointer_type>     &leaf_set_wrt_current_node);

  /**
   * Find a block cluster node in the leaf set of the current block cluster
   * tree, which is not a subset of any block cluster in the given partition.
   * @param partition the given partition.
   * @param it_for_desired_bc_node the returned iterator which points to the
   * selected block cluster node in the leaf set of the current block cluster
   * tree. N.B. Only when the returned pointer is not \p nullptr, this iterator
   * is meaningful.
   * @return pointer to the selected block cluster node.
   */
  node_pointer_type
  find_leaf_bc_node_not_subset_of_bc_nodes_in_partition(
    const std::vector<node_pointer_type> &partition,
    typename std::vector<node_pointer_type>::const_iterator
      &it_for_desired_bc_node) const;

  /**
   * Find a block cluster node in the leaf set of the current block cluster
   * tree, which does not belong to the partition.
   * @param partition the given partition.
   * @param it_for_desired_bc_node the returned iterator which points to the
   * selected block cluster node in the leaf set of the current block cluster
   * tree. N.B. Only when the returned pointer is not \p nullptr, this iterator
   * is meaningful.
   * @return pointer to the selected block cluster node.
   */
  node_pointer_type
  find_leaf_bc_node_not_in_partition(
    const std::vector<node_pointer_type> &partition,
    typename std::vector<node_pointer_type>::const_iterator
      &it_for_desired_bc_node) const;

  node_pointer_type              root_node;
  std::vector<node_pointer_type> leaf_set;
  std::vector<node_pointer_type> near_field_set;
  std::vector<node_pointer_type> far_field_set;
  unsigned int                   n_min;
  Number                         eta;

  /**
   * Depth of the tree, which is the maximum level plus one.
   *
   * \mynote{The level number starts from 0, which is assigned to the root
   * node. This definition of the block cluster tree depth is one more than
   * the definition in \cite{KriemannParallel2005a}.}
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

  /**
   * Whether the current block cluster tree is a subtree of an existing tree.
   */
  bool is_subtree;
};


template <int spacedim, typename Number>
std::ostream &
operator<<(std::ostream                             &out,
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


/**
 * Split a block cluster node in a block cluster tree according to the given
 * split mode.
 *
 * After the splitting, the total number of block cluster nodes in the \bct
 * will be updated. Whether the leaf set will be added with the newly created
 * \bc
 * nodes is controlled by the flag \p if_add_child_nodes_to_leaf_set.
 *
 * \mynote{If the newly created \bc nodes are to be added to the leaf set,
 * their parent \bcn should firstly be erased from the leaf set. This
 * operation is not included in this function and should be performed
 * beforehand.}
 *
 * @param bc_node Pointer to the block cluster node
 * @param bc_tree Reference to the block cluster tree
 * @param split_mode
 * @param if_add_child_nodes_to_leaf_set If the newly created block cluster
 * nodes will be appended to the leaf set of the block cluster tree.
 */
template <int spacedim, typename Number>
void
split_block_cluster_node(
  TreeNode<BlockCluster<spacedim, Number>,
           BlockClusterTree<spacedim, Number>::child_num> *bc_node,
  BlockClusterTree<spacedim, Number>                      &bc_tree,
  const TreeNodeSplitMode                                  split_mode,
  const bool if_add_child_nodes_to_leaf_set = true)
{
  /**
   * \workflow_start
   */

  // Array of empty child pointers used for initializing a block
  // cluster tree node.
  const std::array<
    typename BlockClusterTree<spacedim, Number>::node_pointer_type,
    BlockClusterTree<spacedim, Number>::child_num>
    empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};

  switch (split_mode)
    {
        case HorizontalSplitMode: {
          /**
           * <ul>
           * <li> Horizontal split mode
           * <ol>
           */

          /**
           * <li> Set the split mode of the current block cluster node.
           */
          bc_node->set_split_mode(HorizontalSplitMode);

          typename ClusterTree<spacedim, Number>::node_pointer_type
            sigma_node_pointer = bc_node->get_data_reference().get_sigma_node();
          Assert(sigma_node_pointer != nullptr, ExcInternalError());

          for (unsigned int i = 0;
               i < (ClusterTree<spacedim, Number>::child_num);
               i++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                tau_son_node_pointer = bc_node->get_data_reference()
                                         .get_tau_node()
                                         ->get_child_pointer(i);
              Assert(tau_son_node_pointer != nullptr, ExcInternalError());

              /**
               * <li> Create a new block cluster node by constructing its
               * index set or index range as \f$\tau^* \times \sigma\f$ with
               * \f$\tau^* \in S(\tau)\f$. Its parent node is set to the
               * current block cluster node.
               */
              typename BlockClusterTree<spacedim, Number>::node_pointer_type
                child_block_cluster_node = CreateTreeNode<
                  typename BlockClusterTree<spacedim, Number>::data_value_type,
                  BlockClusterTree<spacedim, Number>::child_num>(
                  typename BlockClusterTree<spacedim, Number>::data_value_type(
                    tau_son_node_pointer, sigma_node_pointer),
                  bc_node->get_level() + 1,
                  empty_child_pointers,
                  bc_node,
                  UnsplitMode);

              /**
               * <li> Check if the block cluster node is small.
               */
              child_block_cluster_node->get_data_reference()
                .check_is_near_field(bc_tree.n_min);

              /**
               * <li> Append this new node as a child of the current block
               * cluster node.
               */
              bc_node->set_child_pointer(i, child_block_cluster_node);

              if (if_add_child_nodes_to_leaf_set)
                {
                  /**
                   * <li> Add this new node to the leaf set if necessary.
                   */
                  bc_tree.leaf_set.push_back(child_block_cluster_node);
                }

              /**
               * <li> Increase the total number of nodes in the block cluster
               * tree.
               */
              bc_tree.increase_node_num();
            }

          /**
           * </ol>
           */
          break;
        }
        case VerticalSplitMode: {
          /**
           * <li> Vertical split mode
           * <ol>
           */

          /**
           * <li> Set the split mode of the current block cluster node.
           */
          bc_node->set_split_mode(VerticalSplitMode);

          typename ClusterTree<spacedim, Number>::node_pointer_type
            tau_node_pointer = bc_node->get_data_reference().get_tau_node();
          Assert(tau_node_pointer != nullptr, ExcInternalError());

          for (unsigned int i = 0;
               i < (ClusterTree<spacedim, Number>::child_num);
               i++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                sigma_son_node_pointer = bc_node->get_data_reference()
                                           .get_sigma_node()
                                           ->get_child_pointer(i);
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

              /**
               * <li> Create a new block cluster node by constructing its index set
               * as \f$\tau \times \sigma^*\f$ with \f$\sigma^* \in
               * S(\sigma)\f$. Its parent node is set to the current block
               * cluster node.
               */
              typename BlockClusterTree<spacedim, Number>::node_pointer_type
                child_block_cluster_node = CreateTreeNode<
                  typename BlockClusterTree<spacedim, Number>::data_value_type,
                  BlockClusterTree<spacedim, Number>::child_num>(
                  typename BlockClusterTree<spacedim, Number>::data_value_type(
                    tau_node_pointer, sigma_son_node_pointer),
                  bc_node->get_level() + 1,
                  empty_child_pointers,
                  bc_node,
                  UnsplitMode);

              /**
               * <li> Check if the block cluster node is small.
               */
              child_block_cluster_node->get_data_reference()
                .check_is_near_field(bc_tree.n_min);

              /**
               * <li> Append this new node as a child of the current block cluster
               * node.
               */
              bc_node->set_child_pointer(i, child_block_cluster_node);

              if (if_add_child_nodes_to_leaf_set)
                {
                  /**
                   * <li> Add this new node to the leaf set if necessary.
                   */
                  bc_tree.leaf_set.push_back(child_block_cluster_node);
                }

              /**
               * <li> Increase the total number of nodes in the block cluster
               * tree.
               */
              bc_tree.increase_node_num();
            }

          /**
           * </ol>
           */

          break;
        }
        case CrossSplitMode: {
          /**
           * <li> Cross split mode
           * <ol>
           */

          /**
           * <li> Set the split mode of the current block cluster node.
           */
          bc_node->set_split_mode(CrossSplitMode);

          unsigned int child_counter = 0;
          // Iterate over each child of the cluster \f$\tau\f$.
          for (unsigned int i = 0;
               i < (ClusterTree<spacedim, Number>::child_num);
               i++)
            {
              typename ClusterTree<spacedim, Number>::node_pointer_type
                tau_son_node_pointer = bc_node->get_data_pointer()
                                         ->get_tau_node()
                                         ->get_child_pointer(i);
              Assert(tau_son_node_pointer != nullptr, ExcInternalError());

              // Iterate over each child of the cluster \f$\sigma\f$.
              for (unsigned int j = 0;
                   j < (ClusterTree<spacedim, Number>::child_num);
                   j++)
                {
                  typename ClusterTree<spacedim, Number>::node_pointer_type
                    sigma_son_node_pointer = bc_node->get_data_pointer()
                                               ->get_sigma_node()
                                               ->get_child_pointer(j);
                  Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

                  /**
                   * <li> Create a new block cluster node by constructing its
                   * index set as \f$\tau^* \times \sigma^*\f$ with \f$\tau^*
                   * \in S(\tau)\f$ and \f$\sigma^* \in S(\sigma)\f$. Its
                   * parent node is set to the current block cluster node.
                   */
                  typename BlockClusterTree<spacedim, Number>::node_pointer_type
                    child_block_cluster_node = CreateTreeNode<
                      typename BlockClusterTree<spacedim,
                                                Number>::data_value_type,
                      BlockClusterTree<spacedim, Number>::child_num>(
                      typename BlockClusterTree<spacedim, Number>::
                        data_value_type(tau_son_node_pointer,
                                        sigma_son_node_pointer),
                      bc_node->get_level() + 1,
                      empty_child_pointers,
                      bc_node,
                      UnsplitMode);

                  /**
                   * <li> Check if the block cluster node is small.
                   */
                  child_block_cluster_node->get_data_reference()
                    .check_is_near_field(bc_tree.n_min);

                  /**
                   * <li> Append this new node as one of the children of the current
                   * block cluster node.
                   */
                  bc_node->set_child_pointer(child_counter,
                                             child_block_cluster_node);

                  if (if_add_child_nodes_to_leaf_set)
                    {
                      /**
                       * <li> Add this new node to the leaf set if necessary.
                       */
                      bc_tree.leaf_set.push_back(child_block_cluster_node);
                    }

                  /**
                   * <li> Increase the total number of nodes in the block cluster
                   * tree.
                   */
                  bc_tree.increase_node_num();

                  child_counter++;
                }
            }

          /**
           * </ol>
           */
          break;
        }
        default: {
          Assert(false, ExcInternalError());

          break;
        }
    }

  /**
   *   </ul>
   * \workflow_end
   */
}


/**
 * Prune all those \bc nodes which are descendants of the \bc nodes specified
 * in
 * the given \p partition.
 *
 * \mynote{Because the \bc nodes in the given \p partition are not contained in
 * the \bct, it is necessary to traverse the whole \bct by recursion, during
 * which each \bcn is checked whether it belongs to the \p partition so that its
 * descendants should be pruned.}
 *
 * @param bct The \bct to be pruned.
 * @param partition The given partition whose descendants will be pruned from
 * the original \bct.
 * @param bc_node The current \bcn to be checked and determined whether it
 * belongs to the partition so that its descendants should be pruned.
 */
template <int spacedim, typename Number = double>
bool
prune_to_partition_recursion(
  BlockClusterTree<spacedim, Number> &bct,
  const std::vector<TreeNode<BlockCluster<spacedim, Number>,
                             BlockClusterTree<spacedim, Number>::child_num> *>
                                                          &partition,
  TreeNode<BlockCluster<spacedim, Number>,
           BlockClusterTree<spacedim, Number>::child_num> *bc_node)
{
  bool is_pruned = false;

  if (find_pointer_data(partition.begin(), partition.end(), bc_node) !=
      partition.end())
    {
      is_pruned = bct.prune_descendants_from_node(bc_node);
    }
  else
    {
      for (unsigned int i = 0; i < bc_node->get_child_num(); i++)
        {
          is_pruned |=
            prune_to_partition_recursion(bct,
                                         partition,
                                         bc_node->get_child_pointer(i));
        }
    }

  return is_pruned;
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree()
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(1)
  , eta(1.0)
  , depth(0)
  , max_level(-1)
  , node_num(0)
  , is_subtree(false)
{}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  const ClusterTree<spacedim, Number> &TI,
  const ClusterTree<spacedim, Number> &TJ,
  const unsigned int                   n_min)
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(n_min)
  , eta(1.0)
  , depth(0)
  , max_level(-1)
  , node_num(0)
  , is_subtree(false)
{
  if (this->n_min < 1)
    {
      this->n_min = std::min(TI.get_n_min(), TJ.get_n_min());
    }

  // Initialize the four null child pointers.
  const std::array<node_pointer_type,
                   BlockClusterTree<spacedim, Number>::child_num>
    empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
  root_node = CreateTreeNode<data_value_type,
                             BlockClusterTree<spacedim, Number>::child_num>(
    data_value_type(TI.get_root(), TJ.get_root()),
    0,
    empty_child_pointers,
    static_cast<node_pointer_type>(nullptr),
    UnsplitMode);

  depth     = 1;
  max_level = 0;
  node_num  = 1;

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  typename ClusterTree<spacedim, Number>::node_pointer_type tau_root_node,
  typename ClusterTree<spacedim, Number>::node_pointer_type sigma_root_node,
  const unsigned int                                        n_min)
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(n_min)
  , eta(1.0)
  , depth(1)
  , max_level(0)
  , node_num(1)
  , is_subtree(false)
{
  // Initialize the four null child pointers.
  const std::array<node_pointer_type,
                   BlockClusterTree<spacedim, Number>::child_num>
    empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};

  root_node = CreateTreeNode<data_value_type,
                             BlockClusterTree<spacedim, Number>::child_num>(
    data_value_type(tau_root_node, sigma_root_node),
    0,
    empty_child_pointers,
    static_cast<node_pointer_type>(nullptr),
    UnsplitMode);

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(node_pointer_type  bc_node,
                                                     const unsigned int n_min)
  : root_node(bc_node)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(n_min)
  , eta(1.0)
  , depth(1)
  , max_level(0)
  , node_num(1)
  , is_subtree(true)
{
  build_leaf_set();
  categorize_near_and_far_field_sets();
  calc_depth_and_max_level();
  node_num = CountTreeNodes(root_node);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(node_pointer_type  bc_node,
                                                     const Number       eta,
                                                     const unsigned int n_min)
  : root_node(bc_node)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(n_min)
  , eta(eta)
  , depth(1)
  , max_level(0)
  , node_num(1)
  , is_subtree(true)
{
  build_leaf_set();
  categorize_near_and_far_field_sets();
  calc_depth_and_max_level();
  node_num = CountTreeNodes(root_node);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  const ClusterTree<spacedim, Number> &TI,
  const ClusterTree<spacedim, Number> &TJ,
  const Number                         eta,
  const unsigned int                   n_min)
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(n_min)
  , eta(eta)
  , depth(0)
  , max_level(-1)
  , node_num(0)
  , is_subtree(false)
{
  if (this->n_min < 1)
    {
      this->n_min = std::min(TI.get_n_min(), TJ.get_n_min());
    }

  // Initialize the four null child pointers.
  const std::array<node_pointer_type,
                   BlockClusterTree<spacedim, Number>::child_num>
    empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
  root_node = CreateTreeNode<data_value_type,
                             BlockClusterTree<spacedim, Number>::child_num>(
    data_value_type(TI.get_root(), TJ.get_root()),
    0,
    empty_child_pointers,
    static_cast<node_pointer_type>(nullptr),
    UnsplitMode);

  depth     = 1;
  max_level = 0;
  node_num  = 1;

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  const BlockClusterTree<spacedim, Number> &bct)
  : root_node(nullptr)
  , leaf_set(0)
  , near_field_set(0)
  , far_field_set(0)
  , n_min(bct.n_min)
  , eta(bct.eta)
  , depth(bct.depth)
  , max_level(bct.max_level)
  , node_num(bct.node_num)
  , is_subtree(false)
{
  root_node = CopyTree(bct.get_root());

  build_leaf_set();
  categorize_near_and_far_field_sets();
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::BlockClusterTree(
  BlockClusterTree<spacedim, Number> &&bct)
  : root_node(bct.root_node)
  , leaf_set(bct.leaf_set)
  , near_field_set(bct.near_field_set)
  , far_field_set(bct.far_field_set)
  , n_min(bct.n_min)
  , eta(bct.eta)
  , depth(bct.depth)
  , max_level(bct.max_level)
  , node_num(bct.node_num)
  , is_subtree(bct.is_subtree)
{
  bct.clear();
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number> &
BlockClusterTree<spacedim, Number>::operator=(
  const BlockClusterTree<spacedim, Number> &bct)
{
  /**
   * If the current block cluster tree is a subtree of an existing block
   * cluster tree, clear its data members instead of releasing their memory.
   */
  if (is_subtree)
    {
      clear();
    }
  else
    {
      release();
    }

  root_node  = CopyTree(bct.get_root());
  n_min      = bct.n_min;
  eta        = bct.eta;
  depth      = bct.depth;
  max_level  = bct.max_level;
  node_num   = bct.node_num;
  is_subtree = false;

  build_leaf_set();
  categorize_near_and_far_field_sets();

  return (*this);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number> &
BlockClusterTree<spacedim, Number>::operator=(
  BlockClusterTree<spacedim, Number> &&bct)
{
  if (is_subtree)
    {
      clear();
    }
  else
    {
      release();
    }

  root_node      = bct.root_node;
  leaf_set       = bct.leaf_set;
  near_field_set = bct.near_field_set;
  far_field_set  = bct.far_field_set;
  n_min          = bct.n_min;
  eta            = bct.eta;
  depth          = bct.depth;
  max_level      = bct.max_level;
  node_num       = bct.node_num;
  is_subtree     = bct.is_subtree;

  bct.clear();

  return (*this);
}


template <int spacedim, typename Number>
BlockClusterTree<spacedim, Number>::~BlockClusterTree()
{
  if (is_subtree)
    {
      clear();
    }
  else
    {
      release();
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::release()
{
  DeleteTree(root_node);
  clear();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::clear()
{
  root_node = nullptr;
  leaf_set.clear();
  near_field_set.clear();
  far_field_set.clear();
  depth      = 0;
  max_level  = -1;
  node_num   = 0;
  is_subtree = false;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_tensor_product()
{
  partition_tensor_product_from_block_cluster_node(root_node, leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_coarse_non_tensor_product()
{
  partition_coarse_non_tensor_product_from_block_cluster_node(root_node,
                                                              leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_fine_non_tensor_product()
{
  partition_fine_non_tensor_product_from_block_cluster_node(root_node,
                                                            leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering,
  const std::vector<Point<spacedim>> &all_support_points)
{
  partition_from_block_cluster_node(root_node,
                                    internal_to_external_dof_numbering,
                                    all_support_points,
                                    leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<types::global_dof_index>
    &internal_to_external_dof_numbering_I,
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering_J,
  const std::vector<Point<spacedim>> &all_support_points_in_I,
  const std::vector<Point<spacedim>> &all_support_points_in_J)
{
  partition_from_block_cluster_node(root_node,
                                    internal_to_external_dof_numbering_I,
                                    internal_to_external_dof_numbering_J,
                                    all_support_points_in_I,
                                    all_support_points_in_J,
                                    leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering,
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number>          &cell_size_at_dofs)
{
  partition_from_block_cluster_node(root_node,
                                    internal_to_external_dof_numbering,
                                    all_support_points,
                                    cell_size_at_dofs,
                                    leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition(
  const std::vector<types::global_dof_index>
    &internal_to_external_dof_numbering_I,
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering_J,
  const std::vector<Point<spacedim>> &all_support_points_in_I,
  const std::vector<Point<spacedim>> &all_support_points_in_J,
  const std::vector<Number>          &cell_size_at_dofs_in_I,
  const std::vector<Number>          &cell_size_at_dofs_in_J)
{
  partition_from_block_cluster_node(root_node,
                                    internal_to_external_dof_numbering_I,
                                    internal_to_external_dof_numbering_J,
                                    all_support_points_in_I,
                                    all_support_points_in_J,
                                    cell_size_at_dofs_in_I,
                                    cell_size_at_dofs_in_J,
                                    leaf_set);

  categorize_near_and_far_field_sets();

  calc_depth_and_max_level();
}


template <int spacedim, typename Number>
bool
BlockClusterTree<spacedim, Number>::extend_finer_than_partition(
  const std::vector<node_pointer_type> &partition)
{
  /**
   * Iterate over the leaf set of the current block cluster tree and look for
   * a block cluster which is not contained by any block cluster in the given
   * partition.
   */
  // Iterator pointing to the selected block cluster node in the leaf set.
  typename std::vector<node_pointer_type>::const_iterator
    it_for_desired_bc_node;

  /**
   * Flag variable indicating whether the current block cluster tree has
   * actually been extended. If true, the leaf set of the current block
   * cluster tree has been modified during the extension, which needs a
   * further re-categorization into the near field set and the far field set.
   */
  bool is_tree_extended = false;

  /**
   * Enter the loop by selecting a block cluster node from the leaf set which
   * is not contained in any block cluster nodes in the given partition.
   */
  while (node_pointer_type current_block_cluster_node =
           find_leaf_bc_node_not_subset_of_bc_nodes_in_partition(
             partition, it_for_desired_bc_node))
    {
      /**
       * Because the selected block cluster node in the leaf set is about to
       * be split, we delete the current block cluster node from the leaf set
       * here and add its children to the leaf set later on during calling
       * \p split_block_cluster_node.
       */
      leaf_set.erase(it_for_desired_bc_node);

      /**
       * Select a block cluster node in the given partition, whose index set
       * has a nonempty intersection with the index set of the current block
       * cluster node.
       */
      node_pointer_type bc_node_in_partition =
        find_bc_node_in_partition_intersect_current_bc_node(
          current_block_cluster_node, partition);
      /**
       * Because the given partition is a covering of the complete block
       * cluster index set \f$I \times J\f$, such block cluster node must
       * exist. Hence we make an assertion here.
       */
      Assert(bc_node_in_partition != nullptr, ExcInternalError());

      if (current_block_cluster_node->get_data_reference()
            .get_tau_node()
            ->get_data_reference()
            .is_proper_superset(bc_node_in_partition->get_data_reference()
                                  .get_tau_node()
                                  ->get_data_reference()))
        {
          if (current_block_cluster_node->get_data_reference()
                .get_sigma_node()
                ->get_data_reference()
                .is_proper_superset(bc_node_in_partition->get_data_reference()
                                      .get_sigma_node()
                                      ->get_data_reference()))
            {
              if (current_block_cluster_node->get_data_reference()
                    .get_tau_node()
                    ->get_level() <
                  current_block_cluster_node->get_data_reference()
                    .get_sigma_node()
                    ->get_level())
                {
                  /**
                   * Extend the current block cluster node by splitting its
                   * \f$\tau\f$ node, which is horizontal split, then make
                   * Cartesian product with its \f$\sigma\f$ node. The newly
                   * created \bc nodes are appended to the leaf set.
                   */
                  split_block_cluster_node(current_block_cluster_node,
                                           *this,
                                           HorizontalSplitMode,
                                           true);
                }
              else
                {
                  /**
                   * Extend the current block cluster node by splitting its
                   * \f$\sigma\f$ node, which is vertical split, then make
                   * Cartesian product with its \f$\tau\f$ node. The newly
                   * created \bc nodes are appended to the leaf set.
                   */
                  split_block_cluster_node(current_block_cluster_node,
                                           *this,
                                           VerticalSplitMode,
                                           true);
                }
            }
          else
            {
              /**
               * Extend the current block cluster node by splitting its
               * \f$\tau\f$ node, which is horizontal split, then make
               * Cartesian product with its \f$\sigma\f$ node. The newly
               * created \bc nodes are appended to the leaf set.
               */
              split_block_cluster_node(current_block_cluster_node,
                                       *this,
                                       HorizontalSplitMode,
                                       true);
            }
        }
      else
        {
          /**
           * Extend the current block cluster node by splitting its
           * \f$\sigma\f$ node, which is vertical split, then make Cartesian
           * product with its \f$\tau\f$ node. The newly created \bc nodes are
           * appended to the leaf set.
           */
          split_block_cluster_node(current_block_cluster_node,
                                   *this,
                                   VerticalSplitMode,
                                   true);
        }

      /**
       * Update the maximum level and depth of the current tree if necessary.
       */
      if (static_cast<int>(current_block_cluster_node->get_level() + 1) >
          max_level)
        {
          AssertDimension(current_block_cluster_node->get_level(), max_level);

          max_level++;
          depth++;
        }

      is_tree_extended = true;
    }

  /**
   * After block cluster tree extension, re-categorize the near field set and
   * far field set based on the new leaf set. N.B. The value of \p n_min and \p
   * eta does not change. Meanwhile, \p depth and \p max_level have been updated
   * during previous procedures.
   */
  if (is_tree_extended)
    {
      categorize_near_and_far_field_sets();
    }

  return is_tree_extended;
}


template <int spacedim, typename Number>
bool
BlockClusterTree<spacedim, Number>::extend_to_finer_partition(
  const std::vector<node_pointer_type> &partition)
{
  /**
   * Iterate over the leaf set of the current block cluster tree and look for
   * a block cluster which does not belong to the partition.
   */
  // Iterator pointing to the selected block cluster node in the leaf set.
  typename std::vector<node_pointer_type>::const_iterator
    it_for_desired_bc_node;

  // Flag indicating whether the current block cluster tree has actually been
  // extended, thus the leaf set is also modified and needs a
  // re-categorization into the near field set and the far field set.
  bool is_tree_extended = false;

  /**
   * Enter the loop by selecting a block cluster node from the leaf set, which
   * does not belong to the given partition.
   */
  while (
    node_pointer_type current_block_cluster_node =
      find_leaf_bc_node_not_in_partition(partition, it_for_desired_bc_node))
    {
      /**
       * Because the selected block cluster node in the leaf set is about to
       * be split, we delete the current block cluster node from the leaf set
       * here and add its children to the leaf set later on during calling
       * \p split_block_cluster_node.
       */
      leaf_set.erase(it_for_desired_bc_node);

      /**
       * Select a block cluster node in the given partition, whose index set
       * is a proper subset of the index set of the current block cluster
       * node.
       */
      node_pointer_type bc_node_in_partition =
        find_bc_node_in_partition_proper_subset_of_current_bc_node(
          current_block_cluster_node, partition);

      /**
       * Because the partition is a covering of the complete block cluster
       * index set \f$I \times J\f$ and it is finer than the leaf set of the
       * current block cluster tree, such block cluster node must exist. Hence
       * we make an assertion here.
       */
      Assert(bc_node_in_partition != nullptr, ExcInternalError());

      /**
       * Here we ensure that the level difference between \f$\tau\f$ and
       * \f$\sigma\f$ clusters is at most 1.
       */
      if (current_block_cluster_node->get_data_reference()
            .get_tau_node()
            ->get_data_reference()
            .is_proper_superset(bc_node_in_partition->get_data_reference()
                                  .get_tau_node()
                                  ->get_data_reference()))
        {
          if (current_block_cluster_node->get_data_reference()
                .get_sigma_node()
                ->get_data_reference()
                .is_proper_superset(bc_node_in_partition->get_data_reference()
                                      .get_sigma_node()
                                      ->get_data_reference()))
            {
              if (current_block_cluster_node->get_data_reference()
                    .get_tau_node()
                    ->get_level() <
                  current_block_cluster_node->get_data_reference()
                    .get_sigma_node()
                    ->get_level())
                {
                  /**
                   * Extend the current block cluster node by splitting its
                   * \f$\tau\f$ node, which is horizontal split, then make
                   * Cartesian product with its \f$\sigma\f$ node. The newly
                   * created \bc nodes are appended to the leaf set.
                   */
                  split_block_cluster_node(current_block_cluster_node,
                                           *this,
                                           HorizontalSplitMode,
                                           true);
                }
              else
                {
                  /**
                   * Extend the current block cluster node by splitting its
                   * \f$\sigma\f$ node, which is vertical split, then make
                   * Cartesian product with its \f$\tau\f$ node. The newly
                   * created \bc nodes are appended to the leaf set.
                   */
                  split_block_cluster_node(current_block_cluster_node,
                                           *this,
                                           VerticalSplitMode,
                                           true);
                }
            }
          else
            {
              /**
               * Extend the current block cluster node by splitting its
               * \f$\tau\f$ node, which is horizontal split, then make
               * Cartesian product with its \f$\sigma\f$ node. The newly
               * created \bc nodes are appended to the leaf set.
               */
              split_block_cluster_node(current_block_cluster_node,
                                       *this,
                                       HorizontalSplitMode,
                                       true);
            }
        }
      else
        {
          /**
           * Extend the current block cluster node by splitting its
           * \f$\sigma\f$ node, which is vertical split, then make Cartesian
           * product with its \f$\tau\f$ node. The newly created \bc nodes are
           * appended to the leaf set.
           */
          split_block_cluster_node(current_block_cluster_node,
                                   *this,
                                   VerticalSplitMode,
                                   true);
        }

      /**
       * Update the maximum level and depth of the current tree if necessary.
       */
      if (static_cast<int>(current_block_cluster_node->get_level() + 1) >
          max_level)
        {
          AssertDimension(current_block_cluster_node->get_level(), max_level);

          max_level++;
          depth++;
        }

      is_tree_extended = true;
    }

  /**
   * After block cluster tree extension, re-categorize the near field set and
   * far field set based on the new leaf set. N.B. The value of \p n_min and \p
   * eta does not change. Meanwhile, \p depth and \p max_level have been updated
   * during previous procedures.
   */
  if (is_tree_extended)
    {
      categorize_near_and_far_field_sets();
    }

  return is_tree_extended;
}


template <int spacedim, typename Number>
bool
BlockClusterTree<spacedim, Number>::prune_descendants_from_node(
  node_pointer_type bc_node)
{
  bool is_pruned = false;

  /**
   * Delete the subtree emanating from each child node of the current
   * \bcn in the partition.
   */
  for (unsigned int i = 0; i < bc_node->get_child_num(); i++)
    {
      Assert(bc_node->get_child_pointer(i), ExcInternalError());

      is_pruned = true;

      unsigned int node_num_for_deletion =
        CountTreeNodes(bc_node->get_child_pointer(i));
      DeleteTree(bc_node->get_child_pointer(i));
      bc_node->set_child_pointer(i, nullptr);
      decrease_node_num(node_num_for_deletion);
    }

  if (is_pruned)
    {
      /**
       * Update the \bcn status including number of children and split mode.
       */
      bc_node->set_child_num(0);
      bc_node->set_split_mode(UnsplitMode);
    }

  return is_pruned;
}


template <int spacedim, typename Number>
bool
BlockClusterTree<spacedim, Number>::prune_to_partition(
  const std::vector<node_pointer_type> &partition,
  const bool                            is_partition_in_bct)
{
  bool is_pruned = false;

  if (is_partition_in_bct)
    {
      /**
       * If the \bc nodes in the given \p partition are contained in this \bct,
       * we directly prune their descendants.
       */
      for (node_pointer_type bc_node : partition)
        {
          is_pruned |= prune_descendants_from_node(bc_node);
        }
    }
  else
    {
      is_pruned = prune_to_partition_recursion((*this), partition, root_node);
    }

  if (is_pruned)
    {
      build_leaf_set();
      categorize_near_and_far_field_sets();
      calc_depth_and_max_level();
    }

  return is_pruned;
}


template <int spacedim, typename Number>
inline typename BlockClusterTree<spacedim, Number>::node_pointer_type
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
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
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
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
               * Create a new block cluster node as child. Then append this
               * new node as one of the children of the current block cluster
               * node. Finally, recursively partition from this node if the
               * two component clusters have the same indices, i.e. \f$I_1
               * \times I_1\f$ and \f$I_2 \times I_2\f$; otherwise, for \f$I_1
               * \times I_2\f$ and \f$I_2 \times I_1\f$, stop the recursion
               * and directly add them to the leaf set.
               */
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == j)
                {
                  /**
                   * Handle the case for \f$I_1 \times I_1\f$ and \f$I_2
                   * \times I_2\f$.
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
                   * Handle the case for \f$I_1 \times I_2\f$ and \f$I_2
                   * \times I_1\f$. Because the recursion stops here, we need
                   * to check and update its near field property.
                   */
                  child_block_cluster_node->get_data_pointer()->is_small(n_min);
                  leaf_set_wrt_current_node.push_back(child_block_cluster_node);
                }
            }
        }

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the non-tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
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
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
    }
  else
    {
      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);

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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
               * Create a new block cluster node as child. Then append this
               * new node as one of the children of the current block cluster
               * node. Finally, recursively partition from this node using the
               * fine non-tensor product method if the two component clusters
               * have the same indices, i.e. \f$I_1 \times I_1\f$ and \f$I_2
               * \times I_2\f$; for \f$I_1 \times I_2\f$, recursively
               * partition from it using the \f$\mathcal{N}\f$-type partition
               * method; for \f$I_2 \times I_1\f$, recursively partition from
               * it using the \f$\mathcal{N}^*\f$-type partition method.
               */
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == j)
                {
                  /**
                   * Handle the case for \f$I_1 \times I_1\f$ and \f$I_2
                   * \times I_2\f$ by recursively applying the fine non-tensor
                   * product partition method itself.
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
                       * the \f$\mathcal{N}\f$-type recursive partition.
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
                       * the \f$\mathcal{N}^*\f$-type recursive partition.
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
                      /**
                       * This case can never happen.
                       */
                      Assert(false, ExcInternalError());
                    }
                }
            }
        }

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the fine non-tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd><strong>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</strong></dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));
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
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
               * Create a new block cluster node as child. Then append this
               * new node as one of the children of the current block cluster
               * node. Finally, recursively partition from this node if it is
               * on the bottom left corner, i.e. it is the \f$I_2 \times
               * I_1\f$ block cluster; otherwise, for \f$I_1 \times I_1\f$,
               * \f$I_1 \times I_2\f$ and \f$I_2 \times I_2\f$, stop the
               * recursion and directly add them to the leaf set.
               */
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == 1 && j == 0)
                {
                  /**
                   * Handle the case for \f$I_2 \times I_1\f$ and perform the
                   * \f$\mathcal{N}\f$-type recursive partition.
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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the fine non-tensor product \f$\mathcal{N}\f$-type
       * construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
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
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
               * Create a new block cluster node as child. Then append this
               * new node as one of the children of the current block cluster
               * node. Finally, recursively partition from this node if it is
               * on the top right corner, i.e. it is the \f$I_1 \times I_2\f$
               * block cluster; otherwise, for \f$I_1 \times I_1\f$, \f$I_2
               * \times I_1\f$ and \f$I_2 \times I_2\f$, stop the recursion
               * and directly add them to the leaf set.
               */
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              if (i == 0 && j == 1)
                {
                  /**
                   * Handle the case for \f$I_1 \times I_2\f$ and perform the
                   * \f$\mathcal{N}^*\f$-type recursive partition.
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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the fine non-tensor product \f$\mathcal{N}^*\f$-type
       * construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_from_block_cluster_node(
  node_pointer_type current_block_cluster_node,
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering,
  const std::vector<Point<spacedim>> &all_support_points,
  std::vector<node_pointer_type>     &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_admissible_or_small(
        eta, internal_to_external_dof_numbering, all_support_points, n_min))
    {
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              /**
               * Update the total number of nodes in the tree.
               */
              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_from_block_cluster_node(
                child_block_cluster_node,
                internal_to_external_dof_numbering,
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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_from_block_cluster_node(
  node_pointer_type current_block_cluster_node,
  const std::vector<types::global_dof_index>
    &internal_to_external_dof_numbering_I,
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering_J,
  const std::vector<Point<spacedim>> &all_support_points_in_I,
  const std::vector<Point<spacedim>> &all_support_points_in_J,
  std::vector<node_pointer_type>     &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_admissible_or_small(
        eta,
        internal_to_external_dof_numbering_I,
        internal_to_external_dof_numbering_J,
        all_support_points_in_I,
        all_support_points_in_J,
        n_min))
    {
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              /**
               * Update the total number of nodes in the tree.
               */
              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_from_block_cluster_node(
                child_block_cluster_node,
                internal_to_external_dof_numbering_I,
                internal_to_external_dof_numbering_J,
                all_support_points_in_I,
                all_support_points_in_J,
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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      AssertDimension(current_block_cluster_node->get_child_num(),
                      (BlockClusterTree<spacedim, Number>::child_num));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_from_block_cluster_node(
  node_pointer_type current_block_cluster_node,
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering,
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number>          &cell_size_at_dofs,
  std::vector<node_pointer_type>     &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_admissible_or_small(
        eta,
        internal_to_external_dof_numbering,
        all_support_points,
        cell_size_at_dofs,
        n_min))
    {
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_from_block_cluster_node(
                child_block_cluster_node,
                internal_to_external_dof_numbering,
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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      Assert(
        current_block_cluster_node->get_child_num() ==
          (BlockClusterTree<spacedim, Number>::child_num),
        ExcDimensionMismatch(current_block_cluster_node->get_child_num(),
                             (BlockClusterTree<spacedim, Number>::child_num)));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
    }
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::partition_from_block_cluster_node(
  node_pointer_type current_block_cluster_node,
  const std::vector<types::global_dof_index>
    &internal_to_external_dof_numbering_I,
  const std::vector<types::global_dof_index>
                                     &internal_to_external_dof_numbering_J,
  const std::vector<Point<spacedim>> &all_support_points_in_I,
  const std::vector<Point<spacedim>> &all_support_points_in_J,
  const std::vector<Number>          &cell_size_at_dofs_in_I,
  const std::vector<Number>          &cell_size_at_dofs_in_J,
  std::vector<node_pointer_type>     &leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  if (current_block_cluster_node->get_data_pointer()->is_admissible_or_small(
        eta,
        internal_to_external_dof_numbering_I,
        internal_to_external_dof_numbering_J,
        all_support_points_in_I,
        all_support_points_in_J,
        cell_size_at_dofs_in_I,
        cell_size_at_dofs_in_J,
        n_min))
    {
      /**
       * Push back the current cluster node, which is small, to the leaf set
       * and
       * set its split mode as \p UnsplitMode.
       */
      leaf_set_wrt_current_node.push_back(current_block_cluster_node);
      current_block_cluster_node->set_split_mode(UnsplitMode);
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
          Assert(tau_son_node_pointer != nullptr, ExcInternalError());

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
              Assert(sigma_son_node_pointer != nullptr, ExcInternalError());

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
              const std::array<node_pointer_type,
                               BlockClusterTree<spacedim, Number>::child_num>
                empty_child_pointers{{nullptr, nullptr, nullptr, nullptr}};
              node_pointer_type child_block_cluster_node =
                CreateTreeNode<data_value_type,
                               BlockClusterTree<spacedim, Number>::child_num>(
                  data_value_type(tau_son_node_pointer, sigma_son_node_pointer),
                  current_block_cluster_node->get_level() + 1,
                  empty_child_pointers,
                  current_block_cluster_node,
                  UnsplitMode);

              /**
               * Append this new node as one of the children of the current
               * block cluster node.
               */
              current_block_cluster_node->set_child_pointer(
                child_counter, child_block_cluster_node);

              node_num++;
              child_counter++;

              std::vector<node_pointer_type> leaf_set_wrt_child_node;
              partition_from_block_cluster_node(
                child_block_cluster_node,
                internal_to_external_dof_numbering_I,
                internal_to_external_dof_numbering_J,
                all_support_points_in_I,
                all_support_points_in_J,
                cell_size_at_dofs_in_I,
                cell_size_at_dofs_in_J,
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

      /**
       * Make sure the current block cluster have four children, which is
       * ensured by the tensor product construction.
       *
       * <dl class="section note">
       *   <dt>Note</dt>
       *   <dd>The second argument \p BlockClusterTree<spacedim,
       * Number>::child_num should be wrapped between the brackets, otherwise,
       * the program cannot be compiled.</dd>
       * </dl>
       */
      Assert(
        current_block_cluster_node->get_child_num() ==
          (BlockClusterTree<spacedim, Number>::child_num),
        ExcDimensionMismatch(current_block_cluster_node->get_child_num(),
                             (BlockClusterTree<spacedim, Number>::child_num)));

      /**
       * Set the split mode of the current block cluster node as cross.
       */
      current_block_cluster_node->set_split_mode(CrossSplitMode);
    }
}


template <int spacedim, typename Number>
typename BlockClusterTree<spacedim, Number>::node_pointer_type
BlockClusterTree<spacedim, Number>::
  find_leaf_bc_node_not_subset_of_bc_nodes_in_partition(
    const std::vector<node_pointer_type> &partition,
    typename std::vector<node_pointer_type>::const_iterator
      &it_for_desired_bc_node) const
{
  /**
   * <dl class="section">
   *   <dt>Work flow</dt>
   *   <dd>
   */
  node_pointer_type desired_bc_node = nullptr;

  /**
   * Iterate over each block cluster node in the leaf set of the current block
   * cluster tree.
   */
  for (it_for_desired_bc_node = leaf_set.cbegin();
       it_for_desired_bc_node != leaf_set.cend();
       it_for_desired_bc_node++)
    {
      node_pointer_type bc_node_in_leaf_set = (*it_for_desired_bc_node);
      bool is_leaf_bc_node_not_subset_of_bc_nodes_in_partition = true;

      /**
       * Iterate over each block cluster node in the given partition.
       */
      for (node_pointer_type bc_node_in_partition : partition)
        {
          if (bc_node_in_leaf_set->get_data_reference().is_subset(
                bc_node_in_partition->get_data_reference()))
            {
              /**
               * When the index set of the current block cluster node in the
               * leaf set is a subset of the index set of some block cluster
               * node in the given partition, terminate the inner loop and
               * jump to the next leaf node for checking.
               */
              is_leaf_bc_node_not_subset_of_bc_nodes_in_partition = false;

              break;
            }
        }

      if (is_leaf_bc_node_not_subset_of_bc_nodes_in_partition)
        {
          /**
           * Here the desired block cluster node in the leaf set is found.
           * Then
           * exist the outer loop and the iterator \p it_for_desired_bc_node now
           * pointing to this node will also be returned.
           */
          desired_bc_node = bc_node_in_leaf_set;

          break;
        }
    }

  return desired_bc_node;

  /**
   *   </dd>
   * </dl>
   */
}


template <int spacedim, typename Number>
typename BlockClusterTree<spacedim, Number>::node_pointer_type
BlockClusterTree<spacedim, Number>::find_leaf_bc_node_not_in_partition(
  const std::vector<node_pointer_type> &partition,
  typename std::vector<node_pointer_type>::const_iterator
    &it_for_desired_bc_node) const
{
  /**
   * <dl class="section">
   *   <dt>Work flow</dt>
   *   <dd>
   */
  node_pointer_type desired_bc_node = nullptr;

  /**
   * Iterate over each block cluster node in the leaf set of the current block
   * cluster tree.
   */
  for (it_for_desired_bc_node = leaf_set.cbegin();
       it_for_desired_bc_node != leaf_set.cend();
       it_for_desired_bc_node++)
    {
      node_pointer_type bc_node_in_leaf_set = (*it_for_desired_bc_node);
      bool              is_leaf_bc_node_not_in_partition = true;

      /**
       * Iterate over each block cluster node in the given partition.
       */
      for (node_pointer_type bc_node_in_partition : partition)
        {
          if (is_equal(bc_node_in_leaf_set->get_data_reference(),
                       bc_node_in_partition->get_data_reference()))
            {
              /**
               * When the index set of the current block cluster node is equal
               * to the index set of some block cluster node in the given
               * partition, terminate the inner loop and jump to the next leaf
               * node for checking.
               */
              is_leaf_bc_node_not_in_partition = false;

              break;
            }
        }

      if (is_leaf_bc_node_not_in_partition)
        {
          /**
           * Here the desired block cluster node in the leaf set is found.
           * Then
           * exist the outer loop and the iterator \p it_for_desired_bc_node now
           * pointing to this node will also be returned.
           */
          desired_bc_node = bc_node_in_leaf_set;

          break;
        }
    }

  return desired_bc_node;

  /**
   *   </dd>
   * </dl>
   */
}


/**
 * Find a block cluster node in the given partition which has nonempty
 * intersection with the current block cluster node.
 *
 * @param current_bc_node
 * @param partition
 * @return
 */
template <int spacedim, typename Number>
TreeNode<BlockCluster<spacedim, Number>,
         BlockClusterTree<spacedim, Number>::child_num> *
find_bc_node_in_partition_intersect_current_bc_node(
  TreeNode<BlockCluster<spacedim, Number>,
           BlockClusterTree<spacedim, Number>::child_num> *current_bc_node,
  const std::vector<TreeNode<BlockCluster<spacedim, Number>,
                             BlockClusterTree<spacedim, Number>::child_num> *>
    &partition)
{
  using node_pointer_type =
    TreeNode<BlockCluster<spacedim, Number>,
             BlockClusterTree<spacedim, Number>::child_num> *;

  node_pointer_type desired_bc_node = nullptr;

  for (node_pointer_type bc_node_in_partition : partition)
    {
      if (current_bc_node->get_data_reference().has_intersection(
            bc_node_in_partition->get_data_reference()))
        {
          desired_bc_node = bc_node_in_partition;

          break;
        }
    }

  return desired_bc_node;
}


template <int spacedim, typename Number>
TreeNode<BlockCluster<spacedim, Number>,
         BlockClusterTree<spacedim, Number>::child_num> *
find_bc_node_in_partition_proper_subset_of_current_bc_node(
  TreeNode<BlockCluster<spacedim, Number>,
           BlockClusterTree<spacedim, Number>::child_num> *current_bc_node,
  const std::vector<TreeNode<BlockCluster<spacedim, Number>,
                             BlockClusterTree<spacedim, Number>::child_num> *>
    &partition)
{
  using node_pointer_type =
    TreeNode<BlockCluster<spacedim, Number>,
             BlockClusterTree<spacedim, Number>::child_num> *;

  node_pointer_type desired_bc_node = nullptr;

  for (node_pointer_type bc_node_in_partition : partition)
    {
      if (bc_node_in_partition->get_data_reference().is_proper_subset(
            current_bc_node->get_data_reference()))
        {
          desired_bc_node = bc_node_in_partition;

          break;
        }
    }

  return desired_bc_node;
}


template <int spacedim, typename Number>
void
print_block_cluster_node_info_as_dot_node(
  std::ostream &out,
  const TreeNode<BlockCluster<spacedim, Number>,
                 BlockClusterTree<spacedim, Number>::child_num>
    *current_bc_node)
{
  using node_const_pointer_type =
    const TreeNode<BlockCluster<spacedim, Number>,
                   BlockClusterTree<spacedim, Number>::child_num> *;

  /**
   * Create the graph node for the current \bcn. When the current \bcn
   * belongs to the near field, use red background. When it belongs to the far
   * field, use green background. Otherwise, use white background.
   */
  out << "\"" << std::hex << current_bc_node << "\""
      << "[label=<<b>" << std::hex << current_bc_node << "</b><br/>" << std::dec
      << "tau: [";
  print_vector_values(out,
                      current_bc_node->get_data_reference()
                        .get_tau_node()
                        ->get_data_reference()
                        .get_index_range(),
                      ",",
                      false);
  out << ")<br/>sigma: [";
  print_vector_values(out,
                      current_bc_node->get_data_reference()
                        .get_sigma_node()
                        ->get_data_reference()
                        .get_index_range(),
                      ",",
                      false);
  out << ")<br/>Level: " << current_bc_node->get_level() << "<br/>";
  out << "Parent: " << std::hex << current_bc_node->Parent() << ">,";

  std::string node_color;

  if (current_bc_node->get_data_reference().get_is_near_field())
    {
      node_color = "red";
    }
  else if (current_bc_node->get_data_reference().get_is_admissible())
    {
      node_color = "green";
    }
  else
    {
      node_color = "white";
    }
  out << "fillcolor = " << node_color << "]\n\n";

  /**
   * Construct the relationship between the current node and its children.
   */
  for (unsigned int i = 0; i < current_bc_node->get_child_num(); i++)
    {
      node_const_pointer_type child_bc_node =
        current_bc_node->get_child_pointer(i);
      Assert(child_bc_node != nullptr, ExcInternalError());

      out << "\"" << std::hex << current_bc_node << "\""
          << "->"
          << "\"" << std::hex << child_bc_node << "\"\n";
    }
  out << "\n";

  /**
   * Print each child \bcn.
   */
  for (unsigned int i = 0; i < current_bc_node->get_child_num(); i++)
    {
      node_const_pointer_type child_bc_node =
        current_bc_node->get_child_pointer(i);
      Assert(child_bc_node != nullptr, ExcInternalError());

      print_block_cluster_node_info_as_dot_node(out, child_bc_node);
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
std::size_t
BlockClusterTree<spacedim, Number>::memory_consumption() const
{
  return sizeof(*this) +
         (dealii::MemoryConsumption::memory_consumption(leaf_set) -
          sizeof(leaf_set)) +
         (dealii::MemoryConsumption::memory_consumption(near_field_set) -
          sizeof(near_field_set)) +
         (dealii::MemoryConsumption::memory_consumption(far_field_set) -
          sizeof(far_field_set)) +
         memory_consumption_of_all_block_clusters();
}


template <int spacedim, typename Number>
std::size_t
BlockClusterTree<spacedim, Number>::memory_consumption_of_all_block_clusters()
  const
{
  std::size_t memory_size = 0;

  Preorder_for_memory_consumption(root_node, memory_size);

  return memory_size;
}


template <int spacedim, typename Number>
inline std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_leaf_set()
{
  return leaf_set;
}


template <int spacedim, typename Number>
inline const std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_leaf_set() const
{
  return leaf_set;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::build_leaf_set()
{
  leaf_set.clear();

  GetTreeLeaves(root_node, leaf_set);
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::write_leaf_set(std::ostream &out) const
{
  for (node_pointer_type bc_node : leaf_set)
    {
      /**
       * Print index range of cluster \f$\tau\f$.
       */
      out << "[";
      print_vector_values(out,
                          bc_node->get_data_reference()
                            .get_tau_node()
                            ->get_data_reference()
                            .get_index_range(),
                          " ",
                          false);
      out << "),";

      /**
       * Print index range of cluster \f$\sigma\f$.
       */
      out << "[";
      print_vector_values(out,
                          bc_node->get_data_reference()
                            .get_sigma_node()
                            ->get_data_reference()
                            .get_index_range(),
                          " ",
                          false);
      out << "),";

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
  std::ostream                       &out,
  const LAPACKFullMatrixExt<Number1> &matrix,
  const Number1                       singular_value_threshold) const
{
  for (node_pointer_type bc_node : leaf_set)
    {
      const std::array<types::global_dof_index, 2> &tau_index_range =
        bc_node->get_data_reference()
          .get_tau_node()
          ->get_data_reference()
          .get_index_range();
      const std::array<types::global_dof_index, 2> &sigma_index_range =
        bc_node->get_data_reference()
          .get_sigma_node()
          ->get_data_reference()
          .get_index_range();

      /**
       * Print index range of cluster \f$\tau\f$.
       */
      out << "[";
      print_vector_values(out, tau_index_range, " ", false);
      out << "),";

      /**
       * Print index range of cluster \f$\sigma\f$.
       */
      out << "[";
      print_vector_values(out, sigma_index_range, " ", false);
      out << "),";

      /**
       * Print the \p is_near_field flag.
       */
      out << (bc_node->get_data_reference().get_is_near_field() ? 1 : 0) << ",";

      /**
       * Make a local copy of the matrix block and calculate its rank using
       * SVD.
       */
      const size_t nrows = tau_index_range[1] - tau_index_range[0];
      const size_t ncols = sigma_index_range[1] - sigma_index_range[0];
      LAPACKFullMatrixExt<Number1> local_matrix(nrows, ncols);

      for (size_t i = 0; i < nrows; i++)
        {
          for (size_t j = 0; j < ncols; j++)
            {
              local_matrix(i, j) =
                matrix(tau_index_range[0] + i, sigma_index_range[0] + j);
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
void
BlockClusterTree<spacedim, Number>::print_bct_info_as_dot(
  std::ostream &out) const
{
  /**
   * Write the header of the Graphviz dot file.
   */
  out << "#@startdot\n";
  out << "digraph block_cluster_tree {\n";

  /**
   * Define the node style.
   */
  out << "node [style=filled, shape=box]\n";

  /**
   * Add comment nodes.
   */
  out << "\"Non-leaf block\" [fillcolor=white]\n";
  out << "\"Near field block\" [fillcolor=red]\n";
  out << "\"Far field block\" [fillcolor=green]\n";

  print_block_cluster_node_info_as_dot_node(out, root_node);

  /**
   * Finalize the Graphviz dot file.
   */
  out << "}\n";
  out << "#@enddot" << std::endl;
}


template <int spacedim, typename Number>
inline std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_near_field_set()
{
  return near_field_set;
}


template <int spacedim, typename Number>
inline const std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_near_field_set() const
{
  return near_field_set;
}


template <int spacedim, typename Number>
inline std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_far_field_set()
{
  return far_field_set;
}


template <int spacedim, typename Number>
inline const std::vector<
  typename BlockClusterTree<spacedim, Number>::node_pointer_type> &
BlockClusterTree<spacedim, Number>::get_far_field_set() const
{
  return far_field_set;
}


template <int spacedim, typename Number>
inline unsigned int
BlockClusterTree<spacedim, Number>::get_depth() const
{
  return depth;
}


template <int spacedim, typename Number>
void
BlockClusterTree<spacedim, Number>::calc_depth_and_max_level()
{
  depth     = calc_depth(root_node);
  max_level = depth - 1;
}


template <int spacedim, typename Number>
inline int
BlockClusterTree<spacedim, Number>::get_max_level() const
{
  return max_level;
}


template <int spacedim, typename Number>
inline unsigned int
BlockClusterTree<spacedim, Number>::get_node_num() const
{
  return node_num;
}


template <int spacedim, typename Number>
inline void
BlockClusterTree<spacedim, Number>::set_node_num(unsigned int node_num)
{
  this->node_num = node_num;
}


template <int spacedim, typename Number>
inline unsigned int
BlockClusterTree<spacedim, Number>::increase_node_num(
  unsigned int increased_node_num)
{
  node_num += increased_node_num;

  return node_num;
}


template <int spacedim, typename Number>
inline unsigned int
BlockClusterTree<spacedim, Number>::decrease_node_num(
  unsigned int decreased_node_num)
{
  Assert(node_num >= decreased_node_num, ExcInternalError());

  node_num -= decreased_node_num;

  return node_num;
}


template <int spacedim, typename Number>
inline unsigned int
BlockClusterTree<spacedim, Number>::get_n_min() const
{
  return n_min;
}


template <int spacedim, typename Number>
inline Number
BlockClusterTree<spacedim, Number>::get_eta() const
{
  return eta;
}

/**
 * @}
 */

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BLOCK_CLUSTER_TREE_H_
