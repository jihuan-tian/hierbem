#ifndef INCLUDE_CLUSTER_TREE_H_
#define INCLUDE_CLUSTER_TREE_H_

#include "cluster.h"
#include "debug_tools.h"
#include "tree.h"

/**
 * \brief Class for cluster tree.
 *
 * A cluster tree is a binary tree which holds a hierarchy of linked nodes with
 * the type BinaryTreeNode.
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
  operator<<(std::ostream &                         out,
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
   * Constructor from a full index set and associated support point coordinates.
   *
   * This constructor will create the root node of the cluster tree based on the
   * given data. There is no mesh cell size correction for the cluster diameter.
   * @param index_set The full DoF index set, which will be assigned to the root node.
   * @param all_support_points All the support points.
   */
  ClusterTree(const std::vector<types::global_dof_index> &index_set,
              const std::vector<Point<spacedim>> &        all_support_points,
              const unsigned int                          n_min);

  /**
   * Constructor from a full index set and associated support point coordinates.
   *
   * This constructor will create the root node of the cluster tree based on the
   * given data. There is mesh cell size correction for the cluster diameter.
   * @param index_set The full DoF index set, which will be assigned to the root node.
   * @param all_support_points All the support points.
   */
  ClusterTree(const std::vector<types::global_dof_index> &index_set,
              const std::vector<Point<spacedim>> &        all_support_points,
              const std::vector<Number> &                 cell_size_at_dofs,
              const unsigned int                          n_min);

  /**
   * Destructor which recursively destroys every node in the cluster tree.
   */
  ~ClusterTree();

  /**
   * Perform a recursive partition by starting from the root node.
   *
   * There is no mesh cell size correction to the cluster diameter and cluster
   * pair distance.
   */
  void
  partition(const std::vector<Point<spacedim>> &all_support_points);

  /**
   * Perform a recursive partition by starting from the root node.
   *
   * There is mesh cell size correction to the cluster diameter and cluster
   * pair distance.
   */
  void
  partition(const std::vector<Point<spacedim>> &all_support_points,
            const std::vector<Number> &         cell_size_at_dofs);

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

private:
  /**
   * Perform a recursive partition by starting from a cluster node.
   *
   * There is no mesh cell size correction to the cluster diameter and cluster
   * pair distance.
   * @param all_support_points All the support points.
   */
  void
  partition_from_cluster_node(
    node_pointer_type                   current_cluster_node,
    const std::vector<Point<spacedim>> &all_support_points,
    std::vector<node_pointer_type> &    leaf_set_wrt_current_node);

  /**
   * Perform a recursive partition by starting from a cluster node.
   *
   * There is mesh cell size correction to the cluster diameter and cluster
   * pair distance.
   * @param all_support_points All the support points.
   */
  void
  partition_from_cluster_node(
    node_pointer_type                   current_cluster_node,
    const std::vector<Point<spacedim>> &all_support_points,
    const std::vector<Number> &         cell_size_at_dofs,
    std::vector<node_pointer_type> &    leaf_set_wrt_current_node);

  node_pointer_type              root_node;
  std::vector<node_pointer_type> leaf_set;

  /**
   * Depth of the tree, which is the maximum level plus one.
   */
  unsigned int depth;

  /**
   * Maximum level of the cluster tree, which is \p depth - 1.
   */
  int max_level;

  /**
   * Minimum cluster size, which is used as the condition for stopping box
   * division.
   */
  const unsigned int n_min;

  /**
   * Total number of clusters in the tree.
   */
  unsigned int node_num;
};

template <int spacedim, typename Number>
std::ostream &
operator<<(std::ostream &out, const ClusterTree<spacedim, Number> &cluster_tree)
{
  out << "* Tree depth: " << cluster_tree.get_depth() << "\n";
  out << "* Tree max level: " << cluster_tree.get_max_level() << "\n";
  out << "* Total number of cluster tree nodes obtained during partition: "
      << cluster_tree.get_node_num() << "\n";
  out << "* Total number of cluster tree nodes by recursive tree traversal: "
      << CountTreeNodes(cluster_tree.get_root()) << "\n";

  out << "* Tree nodes:\n";
  PrintTree(out, cluster_tree.root_node);

  out << "* Leaf set: " << cluster_tree.get_leaf_set().size() << " clusters\n";
  print_vector_of_tree_node_pointer_values(out,
                                           cluster_tree.get_leaf_set(),
                                           "\n");

  return out;
}

template <int spacedim, typename Number>
ClusterTree<spacedim, Number>::ClusterTree()
  : root_node(nullptr)
  , leaf_set(0)
  , depth(0)
  , max_level(-1)
  , n_min(2)
  , node_num(0)
{}

template <int spacedim, typename Number>
ClusterTree<spacedim, Number>::ClusterTree(
  const std::vector<types::global_dof_index> &index_set,
  const std::vector<Point<spacedim>> &        all_support_points,
  const unsigned int                          n_min)
  : root_node(nullptr)
  , leaf_set(0)
  , depth(0)
  , max_level(-1)
  , n_min(n_min)
  , node_num(0)
{
  root_node = CreateTreeNode<data_value_type>(
    Cluster<spacedim, Number>(index_set, all_support_points),
    0,
    nullptr,
    nullptr,
    nullptr);

  depth     = 1;
  max_level = 0;
  node_num  = 1;

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}

template <int spacedim, typename Number>
ClusterTree<spacedim, Number>::ClusterTree(
  const std::vector<types::global_dof_index> &index_set,
  const std::vector<Point<spacedim>> &        all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs,
  const unsigned int                          n_min)
  : root_node(nullptr)
  , leaf_set(0)
  , depth(0)
  , max_level(-1)
  , n_min(n_min)
  , node_num(0)
{
  root_node = CreateTreeNode<data_value_type>(
    Cluster<spacedim, Number>(index_set, all_support_points, cell_size_at_dofs),
    0,
    nullptr,
    nullptr,
    nullptr);

  depth     = 1;
  max_level = 0;
  node_num  = 1;

  // Append the only root node to the leaf set.
  leaf_set.push_back(root_node);
}

template <int spacedim, typename Number>
ClusterTree<spacedim, Number>::~ClusterTree()
{
  DeleteTree(root_node);
}

template <int spacedim, typename Number>
void
ClusterTree<spacedim, Number>::partition_from_cluster_node(
  node_pointer_type                   current_cluster_node,
  const std::vector<Point<spacedim>> &all_support_points,
  std::vector<node_pointer_type> &    leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  /**
   * When the size/cardinality of the current cluster is large enough, continue
   * the partition.
   */
  if (current_cluster_node->get_data_pointer()->get_index_set().size() >= n_min)
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
           * N.B. During the creation of the new child cluster, its bounding box
           * will be recalculated, which may be smaller than the child bounding
           * box obtained from the previous bounding box geometric bisection.
           */
          node_pointer_type child_node = CreateTreeNode<data_value_type>(
            Cluster<spacedim, Number>(left_child_index_set, all_support_points),
            current_cluster_node->get_level() + 1,
            nullptr,
            nullptr,
            current_cluster_node);

          /**
           * Append this new node as the left child of the current cluster node.
           */
          current_cluster_node->Left(child_node);
          node_num++;

          std::vector<node_pointer_type> leaf_set_wrt_child_node;
          /**
           * Continue the recursive partition by starting from this child node.
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
           * N.B. During the creation of the new child cluster, its bounding box
           * will be recalculated, which may be smaller than the child bounding
           * box obtained from the previous bounding box geometric bisection.
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
           * Continue the recursive partition by starting from this child node.
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
  const std::vector<Number> &         cell_size_at_dofs,
  std::vector<node_pointer_type> &    leaf_set_wrt_current_node)
{
  leaf_set_wrt_current_node.clear();

  /**
   * When the size/cardinality of the current cluster is large enough, continue
   * the partition.
   */
  if (current_cluster_node->get_data_pointer()->get_index_set().size() >= n_min)
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
           * N.B. During the creation of the new child cluster, its bounding box
           * will be recalculated, which may be smaller than the child bounding
           * box obtained from the previous bounding box geometric bisection.
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
           * Append this new node as the left child of the current cluster node.
           */
          current_cluster_node->Left(child_node);
          node_num++;

          std::vector<node_pointer_type> leaf_set_wrt_child_node;
          /**
           * Continue the recursive partition by starting from this child node.
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
           * N.B. During the creation of the new child cluster, its bounding box
           * will be recalculated, which may be smaller than the child bounding
           * box obtained from the previous bounding box geometric bisection.
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
           * Continue the recursive partition by starting from this child node.
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
ClusterTree<spacedim, Number>::partition(
  const std::vector<Point<spacedim>> &all_support_points)
{
  partition_from_cluster_node(root_node, all_support_points, leaf_set);

  depth     = calc_depth(root_node);
  max_level = depth - 1;
}

template <int spacedim, typename Number>
void
ClusterTree<spacedim, Number>::partition(
  const std::vector<Point<spacedim>> &all_support_points,
  const std::vector<Number> &         cell_size_at_dofs)
{
  partition_from_cluster_node(root_node,
                              all_support_points,
                              cell_size_at_dofs,
                              leaf_set);

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

#endif /* INCLUDE_CLUSTER_TREE_H_ */
