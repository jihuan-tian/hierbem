#ifndef INCLUDE_CLUSTER_TREE_H_
#define INCLUDE_CLUSTER_TREE_H_

#include "cluster.h"
#include "debug_tools.hcu"
#include "tree.h"

namespace IdeoBEM
{
  /**
   * \brief Class for cluster tree.
   *
   * A cluster tree is a binary tree which holds a hierarchy of linked nodes
   * with the type BinaryTreeNode.
   */
  template <int spacedim, typename Number = double>
  class ClusterTree
  {
  public:
    /**
     * Print a whole cluster tree using recursion.
     * @param out
     * @param cluster_tree
     * @return
     */
    template <int spacedim1, typename Number1>
    friend std::ostream &
    operator<<(std::ostream                          &out,
               const ClusterTree<spacedim1, Number1> &cluster_tree);

    /**
     * Data type for a node in the ClusterTree.
     */
    typedef BinaryTreeNode<Cluster<spacedim, Number>> node_value_type;
    /**
     * Pointer type for a node in the ClusterTree.
     */
    typedef BinaryTreeNode<Cluster<spacedim, Number>> *node_pointer_type;
    /**
     * Const pointer type for a node in the ClusterTree.
     */
    typedef const BinaryTreeNode<Cluster<spacedim, Number>>
      *node_const_pointer_type;
    /**
     * Reference type for a node in the ClusterTree.
     */
    typedef BinaryTreeNode<Cluster<spacedim, Number>> &node_reference_type;
    /**
     * Const reference type for a node in the ClusterTree.
     */
    typedef const BinaryTreeNode<Cluster<spacedim, Number>>
      &node_const_reference_type;

    /**
     * Data type for the content held by a node in the ClusterTree.
     */
    typedef Cluster<spacedim, Number> data_value_type;
    /**
     * Pointer type for the content held by a node in the ClusterTree.
     */
    typedef Cluster<spacedim, Number> *data_pointer_type;
    /**
     * Const pointer type for the content held by a node in the ClusterTree.
     */
    typedef const Cluster<spacedim, Number> *data_const_pointer_type;
    /**
     * Reference type for the content held by a node in the ClusterTree.
     */
    typedef Cluster<spacedim, Number> &data_reference_type;
    /**
     * Const reference type for the content held by a node in the ClusterTree.
     */
    typedef const Cluster<spacedim, Number> &data_const_reference_type;

    /**
     * Number of children in a cluster tree.
     *
     * At present, only binary tree is allowed.
     */
    static const unsigned int child_num = 2;

    /**
     * Default constructor, which initializes an empty binary tree.
     */
    ClusterTree();

    /**
     * Construct from only an index set without support point and the partition
     * will be only based on the cardinality of the index set.
     * @param index_set
     * @param n_min
     */
    ClusterTree(const std::vector<types::global_dof_index> &index_set,
                const unsigned int                          n_min);

    /**
     * Constructor from a full index set and associated support point
     * coordinates.
     *
     * This constructor will create the root node of the cluster tree based on
     * the given data. There is no mesh cell size correction for the cluster
     * diameter.
     *
     * @param index_set The full DoF index set, which will be assigned to the root node.
     * @param all_support_points All the support points.
     * @param n_min
     */
    ClusterTree(const std::vector<types::global_dof_index> &index_set,
                const std::vector<Point<spacedim>>         &all_support_points,
                const unsigned int                          n_min);

    /**
     * Constructor from a full index set and associated support point
     * coordinates.
     *
     * This constructor will create the root node of the cluster tree based on
     * the given data. There is mesh cell size correction for the cluster
     * diameter.
     *
     * @param index_set The full DoF index set, which will be assigned to the root node.
     * @param all_support_points All the support points.
     * @param cell_size_at_dofs
     * @param n_min
     */
    ClusterTree(const std::vector<types::global_dof_index> &index_set,
                const std::vector<Point<spacedim>>         &all_support_points,
                const std::vector<Number>                  &cell_size_at_dofs,
                const unsigned int                          n_min);

    /**
     * Copy constructor.
     * @param cluster_tree
     */
    ClusterTree(const ClusterTree<spacedim, Number> &cluster_tree);

    /**
     * Destructor which recursively destroys every node in the cluster tree.
     */
    ~ClusterTree();

    /**
     * Release the memory and status of the cluster tree hierarchy.
     */
    void
    release();

    /**
     * Deep assignment operator.
     *
     * @param cluster_tree
     * @return
     */
    ClusterTree<spacedim, Number> &
    operator=(const ClusterTree<spacedim, Number> &cluster_tree);

    /**
     * Shallow assignment operator.
     *
     * @param cluster_tree
     * @return
     */
    ClusterTree<spacedim, Number> &
    operator=(ClusterTree<spacedim, Number> &&cluster_tree);

    /**
     * Perform a pure cardinality based recursive partition, which will
     * ultimately be used in constructing an \f$\mathcal{H}^p\f$ matrix for
     * example.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>
     * 1. If the initial complete cluster index set \f$I\f$ is sorted, which is
     * usually \f$[0, 1, \cdots, N]\f$, the cardinality based cluster partition
     * produces cluster index sets following the same order, i.e. the
     * cardinality based partition is order preserving.
     * 2. If the initial complete cluster index set \f$I\f$ is also continuous,
     * i.e. it is a continuous integer array, the cardinality based cluster
     * partition also produces continuous cluster index sets. Hence, the
     * cardinality based partition is continuity preserving.
     *   </dd>
     * </dl>
     */
    void
    partition();

    /**
     * Perform a recursive partition dependent on the coordinates of DoF support
     * points by starting from the root node.
     *
     * In this version, there is no mesh cell size correction to the cluster
     * diameter and cluster pair distance.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>
     * 1. If the initial complete cluster index set \f$I\f$ is sorted, which is
     * usually \f$[0, 1, \cdots, N]\f$, the support point coordinates based
     * partition is also order preserving. This is because the two child
     * clusters of the current cluster are built by scanning the index set of
     * the current cluster from beginning to end.
     * 2. The support point coordinates based partition is not continuity
     * preserving.
     *   </dd>
     * </dl>
     */
    void
    partition(const std::vector<Point<spacedim>> &all_support_points);

    /**
     * Perform a recursive partition dependent on the coordinates of DoF support
     * points by starting from the root node.
     *
     * In this version, there is mesh cell size correction to the cluster
     * diameter and cluster pair distance.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>
     * 1. If the initial complete cluster index set \f$I\f$ is sorted, which is
     * usually \f$[0, 1, \cdots, N]\f$, the support point coordinates based
     * partition is also order preserving. This is because the two child
     * clusters of the current cluster are built by scanning the index set of
     * the current cluster from beginning to end.
     * 2. The support point coordinates based partition is not continuity
     * preserving.
     *   </dd>
     * </dl>
     */
    void
    partition(const std::vector<Point<spacedim>> &all_support_points,
              const std::vector<Number>          &cell_size_at_dofs);

    /**
     * Get the pointer to the root node of the cluster tree.
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
     * Get the reference to the internal-to-external DoF numbering.
     *
     * @return
     */
    std::vector<types::global_dof_index> &
    get_internal_to_external_dof_numbering();

    /**
     * Get the reference to the internal-to-external DoF numbering (const
     * version).
     *
     * @return
     */
    const std::vector<types::global_dof_index> &
    get_internal_to_external_dof_numbering() const;

    /**
     * Get the reference to the external-to-internal DoF numbering.
     *
     * @return
     */
    std::vector<types::global_dof_index> &
    get_external_to_internal_dof_numbering();

    /**
     * Get the reference to the external-to-internal DoF numbering (const
     * version).
     *
     * @return
     */
    const std::vector<types::global_dof_index> &
    get_external_to_internal_dof_numbering() const;

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
     * Get the maximum tree level.
     */
    unsigned int
    get_max_level() const;

    /**
     * Get the total number of clusters in the tree.
     */
    unsigned int
    get_node_num() const;

    /**
     * Print the \bct hierarchy information in the directional graph in GraphViz
     * dot format.
     *
     * @param out
     */
    void
    print_tree_info_as_dot(std::ostream &out) const;

    /**
     * Return the flag indicating whether the index sets have been cleared.
     *
     * @return
     */
    bool
    is_index_sets_cleared() const
    {
      return index_sets_cleared;
    }

  private:
    /**
     * Print the \bct hierarchy information recursively as a node in the
     * directional graph in GraphViz dot format.
     *
     * @param out
     */
    void
    _print_tree_info_as_dot_node(std::ostream           &out,
                                 node_const_pointer_type cluster_node) const;

    /**
     * Perform a pure cardinality based recursive partition by starting from a
     * cluster node.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>
     * 1. If the initial complete cluster index set \f$I\f$ is sorted, which is
     * usually \f$[0, 1, \cdots, N]\f$, the cardinality based cluster partition
     * produces cluster index sets following the same order, i.e. the
     * cardinality based partition is order preserving.
     * 2. If the initial complete cluster index set \f$I\f$ is also continuous,
     * i.e. it is a continuous integer array, the cardinality based cluster
     * partition also produces continuous cluster index sets. Hence, the
     * cardinality based partition is continuity preserving.
     *   </dd>
     * </dl>
     *
     * @param current_cluster_node
     * @param leaf_set_wrt_current_node
     */
    void
    partition_from_cluster_node(
      node_pointer_type               current_cluster_node,
      std::vector<node_pointer_type> &leaf_set_wrt_current_node);

    /**
     * Perform a recursive partition dependent on the coordinates of DoF support
     * points by starting from a cluster node.
     *
     * In this version, there is no mesh cell size correction to the cluster
     * diameter and cluster pair distance.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>
     * 1. If the initial complete cluster index set \f$I\f$ is sorted, which is
     * usually \f$[0, 1, \cdots, N]\f$, the support point coordinates based
     * partition is also order preserving. This is because the two child
     * clusters of the current cluster are built by scanning the index set of
     * the current cluster from beginning to end.
     * 2. The support point coordinates based partition is not continuity
     * preserving.
     *   </dd>
     * </dl>
     *
     * @param all_support_points All the support points.
     */
    void
    partition_from_cluster_node(
      node_pointer_type                   current_cluster_node,
      const std::vector<Point<spacedim>> &all_support_points,
      std::vector<node_pointer_type>     &leaf_set_wrt_current_node);

    /**
     * Perform a recursive partition dependent on the coordinates of DoF support
     * points by starting from a cluster node.
     *
     * In this version, there is mesh cell size correction to the cluster
     * diameter and cluster pair distance.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>
     * 1. If the initial complete cluster index set \f$I\f$ is sorted, which is
     * usually \f$[0, 1, \cdots, N]\f$, the support point coordinates based
     * partition is also order preserving. This is because the two child
     * clusters of the current cluster are built by scanning the index set of
     * the current cluster from beginning to end.
     * 2. The support point coordinates based partition is not continuity
     * preserving.
     *   </dd>
     * </dl>
     *
     * @param all_support_points All the support points.
     */
    void
    partition_from_cluster_node(
      node_pointer_type                   current_cluster_node,
      const std::vector<Point<spacedim>> &all_support_points,
      const std::vector<Number>          &cell_size_at_dofs,
      std::vector<node_pointer_type>     &leaf_set_wrt_current_node);

    /**
     * Build the internal-to-external and external-to-internal DoF numberings by
     * iterating over the leaf set.
     *
     * \mynote{Since the leaf set has been constructed using recursion and
     * whenever we come to a cluster node, its left child node is processed
     * before the right child node, the cluster nodes in the leaf set are
     * naturally ordered.}
     */
    void
    build_internal_and_external_dof_numbering_mappings();

    /**
     * Set the index range (in internal DoF numbering) of each cluster node in
     * the cluster tree from its associated index set based on the mapping from
     * the external numbering to internal numbering. Then, clear the index set
     * (in external DoF numbering) for saving memory.
     */
    void
    set_index_ranges_and_clear_index_sets();

    /**
     * Set index ranges and clear index sets recursively by starting from a
     * cluster node in the cluster tree.
     *
     * @param current_cluster_node
     */
    void
    set_index_ranges_and_clear_index_sets_from_cluster_node(
      node_pointer_type current_cluster_node);

    node_pointer_type              root_node;
    std::vector<node_pointer_type> leaf_set;

    /**
     * The numbering mapping from internal DoF indices to external DoF indices.
     */
    std::vector<types::global_dof_index> internal_to_external_dof_numbering;
    /**
     * The numbering mapping from external DoF indices to internal DoF indices.
     */
    std::vector<types::global_dof_index> external_to_internal_dof_numbering;

    /**
     * Depth of the tree, which is the maximum level plus one.
     *
     * \mynote{The level number starts from 0, which is assigned to the root
     * node.}
     */
    unsigned int depth;

    /**
     * Maximum level of the cluster tree, which is \p depth - 1.
     *
     * \mynote{The level number starts from 0, which is assigned to the root
     * node.}
     */
    int max_level;

    /**
     * Minimum cluster size, which is used as the condition for stopping box
     * division.
     */
    unsigned int n_min;

    /**
     * Total number of clusters in the tree.
     */
    unsigned int node_num;

    /**
     * Whether the index sets for all cluster nodes are cleared.
     */
    bool index_sets_cleared;
  };

  template <int spacedim, typename Number>
  std::ostream &
  operator<<(std::ostream                        &out,
             const ClusterTree<spacedim, Number> &cluster_tree)
  {
    out << "* Tree depth: " << cluster_tree.get_depth() << "\n";
    out << "* Tree max level: " << cluster_tree.get_max_level() << "\n";
    out << "* Total number of cluster tree nodes obtained during partition: "
        << cluster_tree.get_node_num() << "\n";
    out << "* Total number of cluster tree nodes by recursive tree traversal: "
        << CountTreeNodes(cluster_tree.get_root()) << "\n";

    out << "* Tree nodes:\n";
    PrintTree(out, cluster_tree.root_node);

    out << "* Leaf set: " << cluster_tree.get_leaf_set().size()
        << " clusters\n";
    print_vector_of_tree_node_pointer_values(out,
                                             cluster_tree.get_leaf_set(),
                                             "\n");

    out << "* Internal-to-external DoF numbering: [";
    print_vector_values(out,
                        cluster_tree.internal_to_external_dof_numbering,
                        " ",
                        false);
    out << "]\n";

    out << "* External-to-internal DoF numbering: [";
    print_vector_values(out,
                        cluster_tree.external_to_internal_dof_numbering,
                        " ",
                        false);
    out << "]\n";

    return out;
  }

  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number>::ClusterTree()
    : root_node(nullptr)
    , leaf_set(0)
    , internal_to_external_dof_numbering(0)
    , external_to_internal_dof_numbering(0)
    , depth(0)
    , max_level(-1)
    , n_min(2)
    , node_num(0)
    , index_sets_cleared(false)
  {}


  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number>::ClusterTree(
    const std::vector<types::global_dof_index> &index_set,
    const unsigned int                          n_min)
    : root_node(nullptr)
    , leaf_set(0)
    , internal_to_external_dof_numbering(0)
    , external_to_internal_dof_numbering(0)
    , depth(0)
    , max_level(-1)
    , n_min(n_min)
    , node_num(0)
    , index_sets_cleared(false)
  {
    root_node = CreateTreeNode<data_value_type>(
      Cluster<spacedim, Number>(index_set), 0, nullptr, nullptr, nullptr);

    internal_to_external_dof_numbering.resize(index_set.size());
    external_to_internal_dof_numbering.resize(index_set.size());

    depth     = 1;
    max_level = 0;
    node_num  = 1;

    // Append the only root node to the leaf set.
    leaf_set.push_back(root_node);
  }


  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number>::ClusterTree(
    const std::vector<types::global_dof_index> &index_set,
    const std::vector<Point<spacedim>>         &all_support_points,
    const unsigned int                          n_min)
    : root_node(nullptr)
    , leaf_set(0)
    , internal_to_external_dof_numbering(0)
    , external_to_internal_dof_numbering(0)
    , depth(0)
    , max_level(-1)
    , n_min(n_min)
    , node_num(0)
    , index_sets_cleared(false)
  {
    root_node = CreateTreeNode<data_value_type>(
      Cluster<spacedim, Number>(index_set, all_support_points),
      0,
      nullptr,
      nullptr,
      nullptr);

    internal_to_external_dof_numbering.resize(index_set.size());
    external_to_internal_dof_numbering.resize(index_set.size());

    depth     = 1;
    max_level = 0;
    node_num  = 1;

    // Append the only root node to the leaf set.
    leaf_set.push_back(root_node);
  }

  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number>::ClusterTree(
    const std::vector<types::global_dof_index> &index_set,
    const std::vector<Point<spacedim>>         &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs,
    const unsigned int                          n_min)
    : root_node(nullptr)
    , leaf_set(0)
    , internal_to_external_dof_numbering(0)
    , external_to_internal_dof_numbering(0)
    , depth(0)
    , max_level(-1)
    , n_min(n_min)
    , node_num(0)
    , index_sets_cleared(false)
  {
    root_node = CreateTreeNode<data_value_type>(
      Cluster<spacedim, Number>(index_set,
                                all_support_points,
                                cell_size_at_dofs),
      0,
      nullptr,
      nullptr,
      nullptr);

    internal_to_external_dof_numbering.resize(index_set.size());
    external_to_internal_dof_numbering.resize(index_set.size());

    depth     = 1;
    max_level = 0;
    node_num  = 1;

    // Append the only root node to the leaf set.
    leaf_set.push_back(root_node);
  }


  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number>::ClusterTree(
    const ClusterTree<spacedim, Number> &cluster_tree)
    : root_node(nullptr)
    , leaf_set(0)
    , internal_to_external_dof_numbering(
        cluster_tree.internal_to_external_dof_numbering)
    , external_to_internal_dof_numbering(
        cluster_tree.external_to_internal_dof_numbering)
    , depth(cluster_tree.depth)
    , max_level(cluster_tree.max_level)
    , n_min(cluster_tree.n_min)
    , node_num(cluster_tree.node_num)
    , index_sets_cleared(cluster_tree.index_sets_cleared)
  {
    root_node = CopyTree(cluster_tree.get_root());
    build_leaf_set();
  }


  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number>::~ClusterTree()
  {
    release();
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::release()
  {
    DeleteTree(root_node);

    root_node = nullptr;
    leaf_set.clear();
    internal_to_external_dof_numbering.clear();
    external_to_internal_dof_numbering.clear();

    depth              = 0;
    max_level          = -1;
    n_min              = 2;
    node_num           = 0;
    index_sets_cleared = false;
  }


  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number> &
  ClusterTree<spacedim, Number>::operator=(
    const ClusterTree<spacedim, Number> &cluster_tree)
  {
    release();

    root_node = CopyTree(cluster_tree.get_root());
    build_leaf_set();
    internal_to_external_dof_numbering =
      cluster_tree.internal_to_external_dof_numbering;
    external_to_internal_dof_numbering =
      cluster_tree.external_to_internal_dof_numbering;

    depth              = cluster_tree.depth;
    max_level          = cluster_tree.max_level;
    n_min              = cluster_tree.n_min;
    node_num           = cluster_tree.node_num;
    index_sets_cleared = cluster_tree.index_sets_cleared;

    return (*this);
  }


  template <int spacedim, typename Number>
  ClusterTree<spacedim, Number> &
  ClusterTree<spacedim, Number>::operator=(
    ClusterTree<spacedim, Number> &&cluster_tree)
  {
    release();

    root_node = cluster_tree.get_root();
    leaf_set  = cluster_tree.leaf_set;
    internal_to_external_dof_numbering =
      cluster_tree.internal_to_external_dof_numbering;
    external_to_internal_dof_numbering =
      cluster_tree.external_to_internal_dof_numbering;

    depth              = cluster_tree.depth;
    max_level          = cluster_tree.max_level;
    n_min              = cluster_tree.n_min;
    node_num           = cluster_tree.node_num;
    index_sets_cleared = cluster_tree.index_sets_cleared;

    cluster_tree.root_node = nullptr;
    cluster_tree.leaf_set.clear();
    cluster_tree.internal_to_external_dof_numbering.clear();
    cluster_tree.external_to_internal_dof_numbering.clear();

    cluster_tree.depth              = 0;
    cluster_tree.max_level          = -1;
    cluster_tree.n_min              = 2;
    cluster_tree.node_num           = 0;
    cluster_tree.index_sets_cleared = false;

    return (*this);
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::partition_from_cluster_node(
    node_pointer_type               current_cluster_node,
    std::vector<node_pointer_type> &leaf_set_wrt_current_node)
  {
    leaf_set_wrt_current_node.clear();

    /**
     * When the cardinality of the current cluster is large enough, continue
     * the partition.
     */
    if (current_cluster_node->get_data_pointer()->get_index_set().size() >
        n_min)
      {
        /**
         * Declare the two child index sets.
         */
        std::vector<types::global_dof_index> left_child_index_set;
        std::vector<types::global_dof_index> right_child_index_set;

        /**
         * Split the index set of the current node into halves.
         */
        const std::vector<types::global_dof_index> &current_index_set =
          current_cluster_node->get_data_pointer()->get_index_set();

        /**
         * Calculate the splitting index in the middle of the index set, which
         * is to be used for constructing half-closed and half-open
         * subintervals.
         */
        const unsigned int splitting_index =
          (current_index_set.size() - 1) / 2 + 1;

        /**
         * Construct the left child index set.
         */
        for (unsigned int i = 0; i < splitting_index; i++)
          {
            left_child_index_set.push_back(current_index_set.at(i));
          }

        /**
         * Construct the right child index set.
         */
        for (unsigned int i = splitting_index; i < current_index_set.size();
             i++)
          {
            right_child_index_set.push_back(current_index_set.at(i));
          }

        Assert(left_child_index_set.size() > 0,
               ExcLowerRange(left_child_index_set.size(), 1));
        Assert(right_child_index_set.size() > 0,
               ExcLowerRange(right_child_index_set.size(), 1));

        if (left_child_index_set.size() > 0)
          {
            node_pointer_type child_node = CreateTreeNode<data_value_type>(
              Cluster<spacedim, Number>(left_child_index_set),
              current_cluster_node->get_level() + 1,
              nullptr,
              nullptr,
              current_cluster_node);

            /**
             * Append this new node as the left child of the current cluster
             * node.
             */
            current_cluster_node->Left(child_node);
            node_num++;

            std::vector<node_pointer_type> leaf_set_wrt_child_node;
            /**
             * Continue the recursive partition by starting from this child
             * node.
             */
            partition_from_cluster_node(child_node, leaf_set_wrt_child_node);

            /**
             * Merge the leaf set wrt. the child cluster node into the
             * leaf set of the current cluster node.
             */
            for (node_pointer_type cluster_node : leaf_set_wrt_child_node)
              {
                leaf_set_wrt_current_node.push_back(cluster_node);
              }
          }

        if (right_child_index_set.size() > 0)
          {
            node_pointer_type child_node = CreateTreeNode<data_value_type>(
              Cluster<spacedim, Number>(right_child_index_set),
              current_cluster_node->get_level() + 1,
              nullptr,
              nullptr,
              current_cluster_node);

            /**
             * Append this new node as the right child of the current cluster
             * node.
             */
            current_cluster_node->Right(child_node);
            node_num++;

            std::vector<node_pointer_type> leaf_set_wrt_child_node;
            /**
             * Continue the recursive partition by starting from this child
             * node.
             */
            partition_from_cluster_node(child_node, leaf_set_wrt_child_node);

            /**
             * Merge the leaf set wrt. the child cluster node into the
             * leaf set of the current cluster node.
             */
            for (node_pointer_type cluster_node : leaf_set_wrt_child_node)
              {
                leaf_set_wrt_current_node.push_back(cluster_node);
              }
          }
      }
    else
      {
        leaf_set_wrt_current_node.push_back(current_cluster_node);
      }
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::partition_from_cluster_node(
    node_pointer_type                   current_cluster_node,
    const std::vector<Point<spacedim>> &all_support_points,
    std::vector<node_pointer_type>     &leaf_set_wrt_current_node)
  {
    leaf_set_wrt_current_node.clear();

    /**
     * When the size/cardinality of the current cluster is large enough,
     * continue the partition.
     */
    if (current_cluster_node->get_data_pointer()->get_index_set().size() >
        n_min)
      {
        /**
         * Divide the bounding box of the current cluster into halves.
         */
        std::pair<SimpleBoundingBox<spacedim, Number>,
                  SimpleBoundingBox<spacedim, Number>>
          bbox_children = current_cluster_node->get_data_pointer()
                            ->get_bounding_box()
                            .divide_geometrically();

        /**
         * Declare the two child index sets.
         */
        std::vector<types::global_dof_index> left_child_index_set;
        std::vector<types::global_dof_index> right_child_index_set;

        /**
         * Determine to which child index set each support point in the original
         * bounding box belongs to.
         */
        for (auto dof_index :
             current_cluster_node->get_data_pointer()->get_index_set())
          {
            if (bbox_children.first.point_inside(
                  all_support_points.at(dof_index)))
              {
                /**
                 * If the support point associated with the current DoF index
                 * belongs to the left child box, add this DoF index to the left
                 * child index set.
                 */
                left_child_index_set.push_back(dof_index);
              }
            else
              {
                /**
                 * Otherwise, add this DoF index to the right child index set.
                 */
                right_child_index_set.push_back(dof_index);
              }
          }

        if (left_child_index_set.size() > 0)
          {
            /**
             * N.B. During the creation of the new child cluster, its bounding
             * box will be recalculated, which may be smaller than the child
             * bounding box obtained from the previous bounding box geometric
             * bisection.
             */
            node_pointer_type child_node = CreateTreeNode<data_value_type>(
              Cluster<spacedim, Number>(left_child_index_set,
                                        all_support_points),
              current_cluster_node->get_level() + 1,
              nullptr,
              nullptr,
              current_cluster_node);

            /**
             * Append this new node as the left child of the current cluster
             * node.
             */
            current_cluster_node->Left(child_node);
            node_num++;

            std::vector<node_pointer_type> leaf_set_wrt_child_node;
            /**
             * Continue the recursive partition by starting from this child
             * node.
             */
            partition_from_cluster_node(child_node,
                                        all_support_points,
                                        leaf_set_wrt_child_node);

            /**
             * Merge the leaf set wrt. the child cluster node into the
             * leaf set of the current cluster node.
             */
            for (node_pointer_type cluster_node : leaf_set_wrt_child_node)
              {
                leaf_set_wrt_current_node.push_back(cluster_node);
              }
          }

        if (right_child_index_set.size() > 0)
          {
            /**
             * N.B. During the creation of the new child cluster, its bounding
             * box will be recalculated, which may be smaller than the child
             * bounding box obtained from the previous bounding box geometric
             * bisection.
             */
            node_pointer_type child_node = CreateTreeNode<data_value_type>(
              Cluster<spacedim, Number>(right_child_index_set,
                                        all_support_points),
              current_cluster_node->get_level() + 1,
              nullptr,
              nullptr,
              current_cluster_node);

            /**
             * Append this new node as the right child of the current cluster
             * node.
             */
            current_cluster_node->Right(child_node);
            node_num++;

            std::vector<node_pointer_type> leaf_set_wrt_child_node;
            /**
             * Continue the recursive partition by starting from this child
             * node.
             */
            partition_from_cluster_node(child_node,
                                        all_support_points,
                                        leaf_set_wrt_child_node);

            /**
             * Merge the leaf set wrt. the child cluster node into the
             * leaf set of the current cluster node.
             */
            for (node_pointer_type cluster_node : leaf_set_wrt_child_node)
              {
                leaf_set_wrt_current_node.push_back(cluster_node);
              }
          }
      }
    else
      {
        leaf_set_wrt_current_node.push_back(current_cluster_node);
      }
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::partition_from_cluster_node(
    node_pointer_type                   current_cluster_node,
    const std::vector<Point<spacedim>> &all_support_points,
    const std::vector<Number>          &cell_size_at_dofs,
    std::vector<node_pointer_type>     &leaf_set_wrt_current_node)
  {
    leaf_set_wrt_current_node.clear();

    /**
     * When the size/cardinality of the current cluster is large enough,
     * continue the partition.
     */
    if (current_cluster_node->get_data_pointer()->get_index_set().size() >
        n_min)
      {
        /**
         * Divide the bounding box of the current cluster into halves.
         */
        std::pair<SimpleBoundingBox<spacedim, Number>,
                  SimpleBoundingBox<spacedim, Number>>
          bbox_children = current_cluster_node->get_data_pointer()
                            ->get_bounding_box()
                            .divide_geometrically();

        /**
         * Declare the two child index sets.
         */
        std::vector<types::global_dof_index> left_child_index_set;
        std::vector<types::global_dof_index> right_child_index_set;

        /**
         * Determine to which child index set each support point in the original
         * bounding box belongs to.
         */
        for (auto dof_index :
             current_cluster_node->get_data_pointer()->get_index_set())
          {
            if (bbox_children.first.point_inside(
                  all_support_points.at(dof_index)))
              {
                /**
                 * If the support point associated with the current DoF index
                 * belongs to the left child box, add this DoF index to the left
                 * child index set.
                 */
                left_child_index_set.push_back(dof_index);
              }
            else
              {
                /**
                 * Otherwise, add this DoF index to the right child index set.
                 */
                right_child_index_set.push_back(dof_index);
              }
          }

        if (left_child_index_set.size() > 0)
          {
            /**
             * N.B. During the creation of the new child cluster, its bounding
             * box will be recalculated, which may be smaller than the child
             * bounding box obtained from the previous bounding box geometric
             * bisection.
             */
            node_pointer_type child_node = CreateTreeNode<data_value_type>(
              Cluster<spacedim, Number>(left_child_index_set,
                                        all_support_points,
                                        cell_size_at_dofs),
              current_cluster_node->get_level() + 1,
              nullptr,
              nullptr,
              current_cluster_node);

            /**
             * Append this new node as the left child of the current cluster
             * node.
             */
            current_cluster_node->Left(child_node);
            node_num++;

            std::vector<node_pointer_type> leaf_set_wrt_child_node;
            /**
             * Continue the recursive partition by starting from this child
             * node.
             */
            partition_from_cluster_node(child_node,
                                        all_support_points,
                                        cell_size_at_dofs,
                                        leaf_set_wrt_child_node);

            /**
             * Merge the leaf set wrt. the child cluster node into the
             * leaf set of the current cluster node.
             */
            for (node_pointer_type cluster_node : leaf_set_wrt_child_node)
              {
                leaf_set_wrt_current_node.push_back(cluster_node);
              }
          }

        if (right_child_index_set.size() > 0)
          {
            /**
             * N.B. During the creation of the new child cluster, its bounding
             * box will be recalculated, which may be smaller than the child
             * bounding box obtained from the previous bounding box geometric
             * bisection.
             */
            node_pointer_type child_node = CreateTreeNode<data_value_type>(
              Cluster<spacedim, Number>(right_child_index_set,
                                        all_support_points,
                                        cell_size_at_dofs),
              current_cluster_node->get_level() + 1,
              nullptr,
              nullptr,
              current_cluster_node);

            /**
             * Append this new node as the right child of the current cluster
             * node.
             */
            current_cluster_node->Right(child_node);
            node_num++;

            std::vector<node_pointer_type> leaf_set_wrt_child_node;
            /**
             * Continue the recursive partition by starting from this child
             * node.
             */
            partition_from_cluster_node(child_node,
                                        all_support_points,
                                        cell_size_at_dofs,
                                        leaf_set_wrt_child_node);

            /**
             * Merge the leaf set wrt. the child cluster node into the
             * leaf set of the current cluster node.
             */
            for (node_pointer_type cluster_node : leaf_set_wrt_child_node)
              {
                leaf_set_wrt_current_node.push_back(cluster_node);
              }
          }
      }
    else
      {
        leaf_set_wrt_current_node.push_back(current_cluster_node);
      }
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim,
              Number>::build_internal_and_external_dof_numbering_mappings()
  {
    types::global_dof_index internal_index = 0;
    for (node_const_pointer_type leaf_node : leaf_set)
      {
        for (types::global_dof_index external_index :
             leaf_node->get_data_reference().get_index_set())
          {
            internal_to_external_dof_numbering[internal_index] = external_index;
            external_to_internal_dof_numbering[external_index] = internal_index;

            internal_index++;
          }
      }
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::set_index_ranges_and_clear_index_sets()
  {
    Assert(external_to_internal_dof_numbering.size() > 0,
           ExcLowerRange(external_to_internal_dof_numbering.size(), 1));

    set_index_ranges_and_clear_index_sets_from_cluster_node(root_node);
    index_sets_cleared = true;
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::
    set_index_ranges_and_clear_index_sets_from_cluster_node(
      node_pointer_type current_cluster_node)
  {
    Assert(external_to_internal_dof_numbering.size() > 0,
           ExcLowerRange(external_to_internal_dof_numbering.size(), 1));

    std::vector<types::global_dof_index> &index_set =
      current_cluster_node->get_data_reference().get_index_set();
    std::array<types::global_dof_index, 2> &index_range =
      current_cluster_node->get_data_reference().get_index_range();

    if (current_cluster_node->get_child_num() > 0)
      {
        set_index_ranges_and_clear_index_sets_from_cluster_node(
          current_cluster_node->get_child_pointer(0));
        set_index_ranges_and_clear_index_sets_from_cluster_node(
          current_cluster_node->get_child_pointer(1));

        index_range[0] = current_cluster_node->get_child_pointer(0)
                           ->get_data_reference()
                           .get_index_range()[0];
        index_range[1] = current_cluster_node->get_child_pointer(1)
                           ->get_data_reference()
                           .get_index_range()[1];
      }
    else
      {
        // N.B. For the index set in a leaf cluster node, it must be an
        // increasing set of integers.
        index_range[0] = external_to_internal_dof_numbering[index_set[0]];
        index_range[1] =
          external_to_internal_dof_numbering[index_set[index_set.size() - 1]] +
          1;
      }

    index_set.clear();
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::partition()
  {
    partition_from_cluster_node(root_node, leaf_set);
    build_internal_and_external_dof_numbering_mappings();
    set_index_ranges_and_clear_index_sets();

    depth     = calc_depth(root_node);
    max_level = depth - 1;
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::partition(
    const std::vector<Point<spacedim>> &all_support_points)
  {
    partition_from_cluster_node(root_node, all_support_points, leaf_set);
    build_internal_and_external_dof_numbering_mappings();
    set_index_ranges_and_clear_index_sets();

    depth     = calc_depth(root_node);
    max_level = depth - 1;
  }

  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::partition(
    const std::vector<Point<spacedim>> &all_support_points,
    const std::vector<Number>          &cell_size_at_dofs)
  {
    partition_from_cluster_node(root_node,
                                all_support_points,
                                cell_size_at_dofs,
                                leaf_set);
    build_internal_and_external_dof_numbering_mappings();
    set_index_ranges_and_clear_index_sets();

    depth     = calc_depth(root_node);
    max_level = depth - 1;
  }

  template <int spacedim, typename Number>
  typename ClusterTree<spacedim, Number>::node_pointer_type
  ClusterTree<spacedim, Number>::get_root() const
  {
    return root_node;
  }


  template <int spacedim, typename Number>
  std::vector<typename ClusterTree<spacedim, Number>::node_pointer_type> &
  ClusterTree<spacedim, Number>::get_leaf_set()
  {
    return leaf_set;
  }


  template <int spacedim, typename Number>
  const std::vector<typename ClusterTree<spacedim, Number>::node_pointer_type> &
  ClusterTree<spacedim, Number>::get_leaf_set() const
  {
    return leaf_set;
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::build_leaf_set()
  {
    leaf_set.clear();

    GetTreeLeaves(root_node, leaf_set);
  }


  template <int spacedim, typename Number>
  std::vector<types::global_dof_index> &
  ClusterTree<spacedim, Number>::get_internal_to_external_dof_numbering()
  {
    return internal_to_external_dof_numbering;
  }


  template <int spacedim, typename Number>
  const std::vector<types::global_dof_index> &
  ClusterTree<spacedim, Number>::get_internal_to_external_dof_numbering() const
  {
    return internal_to_external_dof_numbering;
  }


  template <int spacedim, typename Number>
  std::vector<types::global_dof_index> &
  ClusterTree<spacedim, Number>::get_external_to_internal_dof_numbering()
  {
    return external_to_internal_dof_numbering;
  }


  template <int spacedim, typename Number>
  const std::vector<types::global_dof_index> &
  ClusterTree<spacedim, Number>::get_external_to_internal_dof_numbering() const
  {
    return external_to_internal_dof_numbering;
  }


  template <int spacedim, typename Number>
  unsigned int
  ClusterTree<spacedim, Number>::get_n_min() const
  {
    return n_min;
  }

  template <int spacedim, typename Number>
  unsigned int
  ClusterTree<spacedim, Number>::get_depth() const
  {
    return depth;
  }

  template <int spacedim, typename Number>
  unsigned int
  ClusterTree<spacedim, Number>::get_max_level() const
  {
    return max_level;
  }


  template <int spacedim, typename Number>
  unsigned int
  ClusterTree<spacedim, Number>::get_node_num() const
  {
    return node_num;
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::print_tree_info_as_dot(std::ostream &out) const
  {
    /**
     * Write the header of the Graphviz dot file.
     */
    out << "#@startdot\n";
    out << "digraph clustertree {\n";

    /**
     * Define the node style.
     */
    out << "node [style=filled, shape=box]\n";

    /**
     * Add comment nodes.
     */
    out << "\"Non-leaf cluster\" [fillcolor=white]\n";
    out << "\"Leaf cluster\" [fillcolor=red]\n";

    _print_tree_info_as_dot_node(out, root_node);

    /**
     * Finalize the Graphviz dot file.
     */
    out << "}\n";
    out << "#@enddot" << std::endl;
  }


  template <int spacedim, typename Number>
  void
  ClusterTree<spacedim, Number>::_print_tree_info_as_dot_node(
    std::ostream           &out,
    node_const_pointer_type cluster_node) const
  {
    /**
     * Create the graph node for the current cluster node.
     */
    out << "\"" << std::hex << cluster_node << "\""
        << "[label=<<b>" << std::hex << cluster_node << "</b><br/>" << std::dec
        << "Level: " << cluster_node->get_level() << "<br/>";
    out << "Index set: [";
    print_vector_values(out,
                        cluster_node->get_data_reference().get_index_set(),
                        ",",
                        false);
    out << "]<br/>";
    out << "Index range: [";
    print_vector_values(out,
                        cluster_node->get_data_reference().get_index_range(),
                        ",",
                        false);
    out << ")<br/>";
    out << "Diameter: " << cluster_node->get_data_reference().get_diameter()
        << ">,";

    std::string node_color;

    if (cluster_node->is_leaf())
      {
        node_color = "red";
      }
    else
      {
        node_color = "white";
      }

    out << "fillcolor = " << node_color << "]\n\n";

    /**
     * Construct the relationship between the current node and its children.
     */
    for (unsigned int i = 0; i < cluster_node->get_child_num(); i++)
      {
        Assert(cluster_node->get_child_pointer(i) != nullptr,
               ExcInternalError());

        out << "\"" << std::hex << cluster_node << "\""
            << "->"
            << "\"" << std::hex << cluster_node->get_child_pointer(i) << "\"\n";
      }

    out << "\n";

    /**
     * Print each child node.
     */
    for (unsigned int i = 0; i < cluster_node->get_child_num(); i++)
      {
        Assert(cluster_node->get_child_pointer(i) != nullptr,
               ExcInternalError());

        _print_tree_info_as_dot_node(out, cluster_node->get_child_pointer(i));
      }
  }
} // namespace IdeoBEM

#endif /* INCLUDE_CLUSTER_TREE_H_ */
