/**
 * \file tree.h
 * \brief Implementation of the classes for binary tree node, general tree node
 * and functions for manipulating the trees constructed from these nodes.
 * \ingroup hierarchical_matrices
 * \date 2021-04-18
 * \author Jihuan Tian
 */

#ifndef TREE_H_
#define TREE_H_

/**
 * \ingroup hierarchical_matrices
 * @{
 */

#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <iostream>
#include <new>
#include <vector>

namespace HierBEM
{
  /**
   * \brief Class for binary tree node.
   *
   * In the implementation of hierarchical matrices, a binary tree node is
   * designed to hold a cluster (Cluster) in a cluster tree (ClusterTree) along
   * with two pointers \p left and \p right pointing to the two children belong to
   * the node itself. The template argument \p T should be assigned the type of
   * the cluster.
   */
  template <typename T>
  class BinaryTreeNode
  {
  public:
    /**
     * Default constructor.
     */
    BinaryTreeNode();

    /**
     * Copy constructor.
     */
    BinaryTreeNode(const BinaryTreeNode &node);

    /**
     * Constructor from the given data.
     *
     * N.B. The data of type \p T will be copied into the created node.
     */
    BinaryTreeNode(const T        &data,
                   unsigned int    level,
                   BinaryTreeNode *left   = nullptr,
                   BinaryTreeNode *right  = nullptr,
                   BinaryTreeNode *parent = nullptr);

    /**
     * Get the pointer to the left child node.
     */
    BinaryTreeNode *
    Left(void) const;

    /**
     * Get the pointer to the right child node.
     */
    BinaryTreeNode *
    Right(void) const;

    /**
     * Get the pointer to the \p index'th child.
     * @param index Index of the child
     * @return
     */
    BinaryTreeNode *
    get_child_pointer(std::size_t index) const;

    /**
     * Get the total number of nonempty children.
     */
    unsigned int
    get_child_num() const;

    /**
     * Set the total number of nonempty children.
     * @param child_num
     */
    void
    set_child_num(const unsigned int child_num);

    /**
     * Increase the total number of children.
     */
    void
    increase_child_num(const unsigned int incr_num = 1);

    /**
     * Decrease the total number of children.
     * @return
     */
    void
    decrease_child_num(const unsigned int decr_num = 1);

    /**
     * Get the pointer to the parent node.
     */
    BinaryTreeNode *
    Parent(void) const;

    /**
     * Set the left child node pointer.
     *
     * N.B. The const pointer type in the argument will be converted to
     * non-const pointer type. In this way, even a const node can be added to
     * the tree.
     */
    void
    Left(const BinaryTreeNode *node_pointer);

    /**
     * Set the right child node pointer.
     *
     * N.B. The const pointer type in the argument will be converted to
     * non-const pointer type. In this way, even a const node can be added to
     * the tree.
     */
    void
    Right(const BinaryTreeNode *node_pointer);

    /**
     * Set the pointer to the \p index'th child.
     * @param index Index of the child
     * @param node_pointer Pointer value to be assigned to the specified child.
     */
    void
    set_child_pointer(std::size_t index, const BinaryTreeNode *node_pointer);

    /**
     * Set the pointer to the parent.
     * @param node_pointer
     */
    void
    Parent(const BinaryTreeNode *node_pointer);

    /**
     * Get the pointer to the node data.
     */
    T *
    get_data_pointer();

    /**
     * Get the pointer to the node data (const version).
     */
    const T *
    get_data_pointer() const;

    /**
     * Get the reference to the node data.
     */
    T &
    get_data_reference();

    /**
     * Get the reference to the node data (const version).
     */
    const T &
    get_data_reference() const;

    /**
     * Get the level of the node in the tree.
     */
    unsigned int
    get_level() const;

    /**
     * Set the level of the node.
     * @param level
     */
    void
    set_level(const unsigned int level);

    /**
     * Return whether the current binary tree node is a leaf.
     * @return
     */
    bool
    is_leaf() const;

    /**
     * Return whether the current binary tree node is the root.
     * @return
     */
    bool
    is_root() const;

    /**
     * Check the equality of two binary tree nodes by comparing the contained
     * data.
     * @param node
     * @return
     */
    bool
    operator==(const BinaryTreeNode<T> &node) const;

  private:
    T            data;
    unsigned int level;

    BinaryTreeNode *left;
    BinaryTreeNode *right;
    BinaryTreeNode *parent;

    /**
     * Total number of nonempty children.
     */
    unsigned int child_num;
  };

  template <typename T>
  BinaryTreeNode<T>::BinaryTreeNode()
    : data()
    , level(0)
    , left(nullptr)
    , right(nullptr)
    , parent(nullptr)
    , child_num(0)
  {}


  template <typename T>
  BinaryTreeNode<T>::BinaryTreeNode(const BinaryTreeNode &node)
    : data(node.data)
    , level(node.level)
    , left(node.left)
    , right(node.right)
    , parent(node.parent)
    , child_num(node.child_num)
  {}


  template <typename T>
  BinaryTreeNode<T>::BinaryTreeNode(const T        &data,
                                    unsigned int    level,
                                    BinaryTreeNode *left,
                                    BinaryTreeNode *right,
                                    BinaryTreeNode *parent)
    : data(data)
    , level(level)
    , left(left)
    , right(right)
    , parent(parent)
    , child_num(0)
  {
    if (left != nullptr)
      {
        child_num++;
      }

    if (right != nullptr)
      {
        child_num++;
      }

    /**
     * Increment the \p child_num of the parent node.
     */
    if (parent != nullptr)
      {
        (parent->child_num)++;
      }
  }


  template <typename T>
  BinaryTreeNode<T> *
  BinaryTreeNode<T>::Left() const
  {
    return left;
  }

  template <typename T>
  BinaryTreeNode<T> *
  BinaryTreeNode<T>::Right() const
  {
    return right;
  }


  template <typename T>
  BinaryTreeNode<T> *
  BinaryTreeNode<T>::get_child_pointer(std::size_t index) const
  {
    AssertIndexRange(index, 2);

    switch (index)
      {
        case 0:
          return Left();

          break;
        case 1:
          return Right();

          break;
      }

    return nullptr;
  }


  template <typename T>
  unsigned int
  BinaryTreeNode<T>::get_child_num() const
  {
    return child_num;
  }


  template <typename T>
  void
  BinaryTreeNode<T>::set_child_num(const unsigned int child_num)
  {
    this->child_num = child_num;
  }


  template <typename T>
  void
  BinaryTreeNode<T>::increase_child_num(const unsigned int incr_num)
  {
    child_num += incr_num;
  }


  template <typename T>
  void
  BinaryTreeNode<T>::decrease_child_num(const unsigned int decr_num)
  {
    if (child_num < decr_num)
      {
        child_num = 0;
      }
    else
      {
        child_num -= decr_num;
      }
  }


  template <typename T>
  void
  BinaryTreeNode<T>::set_child_pointer(std::size_t           index,
                                       const BinaryTreeNode *node_pointer)
  {
    AssertIndexRange(index, 2);

    switch (index)
      {
        case 0:
          Left(node_pointer);

          break;
        case 1:
          Right(node_pointer);

          break;
      }
  }


  template <typename T>
  BinaryTreeNode<T> *
  BinaryTreeNode<T>::Parent() const
  {
    return parent;
  }


  template <typename T>
  void
  BinaryTreeNode<T>::Parent(const BinaryTreeNode *node_pointer)
  {
    parent = const_cast<BinaryTreeNode *>(node_pointer);
  }


  template <typename T>
  void
  BinaryTreeNode<T>::Left(const BinaryTreeNode *node_pointer)
  {
    left = const_cast<BinaryTreeNode *>(node_pointer);
  }

  template <typename T>
  void
  BinaryTreeNode<T>::Right(const BinaryTreeNode *node_pointer)
  {
    right = const_cast<BinaryTreeNode *>(node_pointer);
  }

  template <typename T>
  T *
  BinaryTreeNode<T>::get_data_pointer()
  {
    return &data;
  }


  template <typename T>
  T &
  BinaryTreeNode<T>::get_data_reference()
  {
    return data;
  }


  template <typename T>
  const T &
  BinaryTreeNode<T>::get_data_reference() const
  {
    return data;
  }


  template <typename T>
  const T *
  BinaryTreeNode<T>::get_data_pointer() const
  {
    return &data;
  }

  template <typename T>
  unsigned int
  BinaryTreeNode<T>::get_level() const
  {
    return level;
  }


  template <typename T>
  void
  BinaryTreeNode<T>::set_level(const unsigned int level)
  {
    this->level = level;
  }


  template <typename T>
  bool
  BinaryTreeNode<T>::is_leaf() const
  {
    if (child_num == 0)
      {
        return true;
      }
    else
      {
        return false;
      }
  }


  template <typename T>
  bool
  BinaryTreeNode<T>::is_root() const
  {
    if (parent == nullptr)
      {
        return true;
      }
    else
      {
        return false;
      }
  }


  template <typename T>
  bool
  BinaryTreeNode<T>::operator==(const BinaryTreeNode<T> &node) const
  {
    return (this->data == node.data);
  }

  /**
   * The splitting mode for a tree node, which can be cross split (\p
   * CrossSplitMode), horizontal split (\p HorizontalSplitMode), vertical split
   * (\p VerticalSplitMode) and unsplit (\p UnsplitMode).
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>
   * 1. When a tree node has four children, which are constructed via a
   * tensor product of the children of the \f$\tau\f$ cluster and the children
   * of the \f$\sigma\f$ cluster, it must be cross split.
   *
   * 2. When a tree node has two children, which are constructed via a tensor
   * product of the children of the \f$\tau\f$ cluster and the \f$\sigma\f$
   * cluster, i.e. only the rows of the associated \f$\mathcal{H}\f$-matrix node
   * will be split, then it should be horizontal split.
   *
   * 3. When a tree node has two children, which are constructed via a tensor
   * product of the \f$\tau\f$ cluster and the children of the \f$\sigma\f$
   * cluster, i.e. only the columns of the associated \f$\mathcal{H}\f$-matrix
   * node will be split, then it should be vertical split.
   *   </dd>
   * </dl>
   */
  enum TreeNodeSplitMode
  {
    HorizontalSplitMode,
    VerticalSplitMode,
    CrossSplitMode,
    UnsplitMode
  };

  /**
   * \brief Class for general tree node.
   *
   * This general tree node can contains any but \em fixed number of children.
   * \p
   * T is the type of the data held by the node. \p N is the number of children
   * belong to the node.
   */
  template <typename T, std::size_t N>
  class TreeNode
  {
  public:
    /**
     * Default constructor
     */
    TreeNode();

    /**
     * Constructor from the given data without children.
     */
    TreeNode(const T &data);

    /**
     * Constructor from the given data.
     *
     * N.B. The number of children of the parent node will automatically be
     * incremented, because the current node is associated with this parent.
     */
    TreeNode(const T                         &data,
             unsigned int                     level,
             const std::array<TreeNode *, N> &children,
             TreeNode                        *parent     = nullptr,
             TreeNodeSplitMode                split_mode = UnsplitMode);

    /**
     * Copy constructor.
     */
    TreeNode(const TreeNode &node);

    /**
     * Get the pointer to the \p index'th child.
     * @param index Index of the child
     * @return
     */
    TreeNode *
    get_child_pointer(std::size_t index) const;

    /**
     * Set the pointer to the \p index'th child.
     *
     * N.B. The const pointer type in the argument will be converted to
     * non-const pointer type. In this way, even a const node can be added to
     * the tree.
     * @param index Index of the child
     * @param pointer New pointer value
     */
    void
    set_child_pointer(std::size_t index, const TreeNode *pointer);

    /**
     * Get the total number of nonempty children.
     */
    unsigned int
    get_child_num() const;

    /**
     * Set the total number of nonempty children.
     * @param child_num
     */
    void
    set_child_num(const unsigned int child_num);

    /**
     * Increase the total number of children.
     */
    void
    increase_child_num(const unsigned int incr_num = 1);

    /**
     * Decrease the total number of children.
     * @return
     */
    void
    decrease_child_num(const unsigned int decr_num = 1);

    /**
     * Get the pointer to the parent tree node.
     * @return
     */
    TreeNode *
    Parent(void) const;

    /**
     * Set the pointer to the parent tree node.
     * @param node_pointer
     */
    void
    Parent(const TreeNode *node_pointer);

    /**
     * Get the pointer to the node data.
     */
    T *
    get_data_pointer();

    /**
     * Get the pointer to the node data (const version).
     */
    const T *
    get_data_pointer() const;

    /**
     * Get the reference to the node data.
     */
    T &
    get_data_reference();

    /**
     * Get the reference to the node data (const version).
     */
    const T &
    get_data_reference() const;

    /**
     * Get the level of the node in the tree.
     */
    unsigned int
    get_level() const;

    /**
     * Set the level of the node.
     * @param level
     */
    void
    set_level(const unsigned int level);

    /**
     * Return whether the current tree node is a leaf.
     * @return
     */
    bool
    is_leaf() const;

    /**
     * Return whether the current tree node is the root.
     * @return
     */
    bool
    is_root() const;

    /**
     * Return the split mode of the tree node.
     * @return
     */
    TreeNodeSplitMode
    get_split_mode() const;

    /**
     * Set the split mode of the tree node.
     * @param split_mode
     */
    void
    set_split_mode(TreeNodeSplitMode split_mode);

    /**
     * Check the equality of two tree nodes by comparing the contained
     * data.
     * @param node
     * @return
     */
    bool
    operator==(const TreeNode<T, N> &node) const;

  private:
    T            data;
    unsigned int level;

    std::array<TreeNode *, N> children;
    TreeNode                 *parent;

    /**
     * Total number of nonempty children.
     */
    unsigned int child_num;

    TreeNodeSplitMode split_mode;
  };

  template <typename T, std::size_t N>
  TreeNode<T, N>::TreeNode()
    : data()
    , level(0)
    , parent(nullptr)
    , child_num(0)
    , split_mode(UnsplitMode)
  {
    children.fill(nullptr);
  }

  template <typename T, std::size_t N>
  TreeNode<T, N>::TreeNode(const T &data)
    : data(data)
    , level(0)
    , parent(nullptr)
    , child_num(0)
    , split_mode(UnsplitMode)
  {
    children.fill(nullptr);
  }

  template <typename T, std::size_t N>
  TreeNode<T, N>::TreeNode(const T                         &data,
                           unsigned int                     level,
                           const std::array<TreeNode *, N> &children,
                           TreeNode                        *parent,
                           TreeNodeSplitMode                split_mode)
    : data(data)
    , level(level)
    , children(children)
    , parent(parent)
    , child_num(0)
    , split_mode(split_mode)
  {
    /**
     * Count the total number of nonempty children.
     */
    for (auto child : this->children)
      {
        if (child != nullptr)
          {
            child_num++;
          }
      }

    if (child_num == 4)
      {
        /**
         * When the tree node is split into four children, this is a cross
         * splitting mode. Otherwise, \p split_mode is temporarily kept at \p
         * UnsplitMode, which will further be set from outside.
         */
        split_mode = CrossSplitMode;
      }

    /**
     * Increment the \p child_num of the parent node.
     */
    if (parent != nullptr)
      {
        (parent->child_num)++;
      }
  }

  template <typename T, std::size_t N>
  TreeNode<T, N>::TreeNode(const TreeNode &node)
    : data(node.data)
    , level(node.level)
    , children(node.children)
    , parent(node.parent)
    , child_num(node.child_num)
    , split_mode(node.split_mode)
  {}

  template <typename T, std::size_t N>
  TreeNode<T, N> *
  TreeNode<T, N>::get_child_pointer(std::size_t index) const
  {
    AssertIndexRange(index, N);

    return children.at(index);
  }

  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::set_child_pointer(std::size_t           index,
                                    const TreeNode<T, N> *pointer)
  {
    AssertIndexRange(index, N);

    children.at(index) = const_cast<TreeNode *>(pointer);
  }


  template <typename T, std::size_t N>
  unsigned int
  TreeNode<T, N>::get_child_num() const
  {
    return child_num;
  }


  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::set_child_num(const unsigned int child_num)
  {
    this->child_num = child_num;
  }


  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::increase_child_num(const unsigned int incr_num)
  {
    child_num += incr_num;
  }


  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::decrease_child_num(const unsigned int decr_num)
  {
    if (child_num < decr_num)
      {
        child_num = 0;
      }
    else
      {
        child_num -= decr_num;
      }
  }


  template <typename T, std::size_t N>
  TreeNode<T, N> *
  TreeNode<T, N>::Parent() const
  {
    return parent;
  }

  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::Parent(const TreeNode *node_pointer)
  {
    parent = const_cast<TreeNode *>(node_pointer);
  }

  template <typename T, std::size_t N>
  T *
  TreeNode<T, N>::get_data_pointer()
  {
    return &data;
  }

  template <typename T, std::size_t N>
  const T *
  TreeNode<T, N>::get_data_pointer() const
  {
    return &data;
  }


  template <typename T, std::size_t N>
  T &
  TreeNode<T, N>::get_data_reference()
  {
    return data;
  }


  template <typename T, std::size_t N>
  const T &
  TreeNode<T, N>::get_data_reference() const
  {
    return data;
  }


  template <typename T, std::size_t N>
  unsigned int
  TreeNode<T, N>::get_level() const
  {
    return level;
  }


  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::set_level(const unsigned int level)
  {
    this->level = level;
  }


  template <typename T, std::size_t N>
  bool
  TreeNode<T, N>::is_leaf() const
  {
    if (child_num == 0)
      {
        return true;
      }
    else
      {
        return false;
      }
  }


  template <typename T, std::size_t N>
  bool
  TreeNode<T, N>::is_root() const
  {
    if (parent == nullptr)
      {
        return true;
      }
    else
      {
        return false;
      }
  }


  template <typename T, std::size_t N>
  TreeNodeSplitMode
  TreeNode<T, N>::get_split_mode() const
  {
    return split_mode;
  }


  template <typename T, std::size_t N>
  void
  TreeNode<T, N>::set_split_mode(TreeNodeSplitMode split_mode)
  {
    this->split_mode = split_mode;
  }


  template <typename T, std::size_t N>
  bool
  TreeNode<T, N>::operator==(const TreeNode<T, N> &node) const
  {
    return (this->data == node.data);
  }


  /**
   * Create a new binary tree node from the provided data.
   */
  template <typename T>
  BinaryTreeNode<T> *
  CreateTreeNode(const T           &data,
                 unsigned int       level  = 0,
                 BinaryTreeNode<T> *left   = nullptr,
                 BinaryTreeNode<T> *right  = nullptr,
                 BinaryTreeNode<T> *parent = nullptr)
  {
    BinaryTreeNode<T> *p = nullptr;
    p = new BinaryTreeNode<T>(data, level, left, right, parent);

    if (p == nullptr)
      {
        throw(std::bad_alloc());
      }
    else
      {
        return p;
      }
  }

  /**
   * Create a new tree node from the provided data.
   */
  template <typename T, std::size_t N>
  TreeNode<T, N> *
  CreateTreeNode(const T                               &data,
                 unsigned int                           level,
                 const std::array<TreeNode<T, N> *, N> &children,
                 TreeNode<T, N>                        *parent = nullptr,
                 TreeNodeSplitMode split_mode                  = UnsplitMode)
  {
    TreeNode<T, N> *p = nullptr;
    p = new TreeNode<T, N>(data, level, children, parent, split_mode);

    if (p == nullptr)
      {
        throw(std::bad_alloc());
      }
    else
      {
        return p;
      }
  }

  /**
   * Destroy a binary tree node.
   */
  template <typename T>
  void
  DeleteTreeNode(const BinaryTreeNode<T> *p)
  {
    if (p != nullptr)
      {
        delete p;
      }
  }

  /**
   * Destroy a tree node.
   */
  template <typename T, std::size_t N>
  void
  DeleteTreeNode(const TreeNode<T, N> *p)
  {
    if (p != nullptr)
      {
        delete p;
      }
  }

  /**
   * Print the data contained in a binary tree node.
   */
  template <typename T>
  void
  PrintTreeNode(std::ostream &out, const BinaryTreeNode<T> *p)
  {
    out << "Level " << p->get_level() << "\n";
    out << "Number of children: " << p->get_child_num() << "\n";

    out << "Node data:\n" << p->get_data_reference() << "\n";

    if (p->Parent() != nullptr)
      {
        out << "Parent node data: " << p->Parent()->get_data_reference()
            << "\n";
      }
    else
      {
        out << "Parent node data: none\n";
      }

    out << "------------------" << std::endl;
  }

  /**
   * Print the data contained in a tree node.
   */
  template <typename T, std::size_t N>
  void
  PrintTreeNode(std::ostream &out, const TreeNode<T, N> *p)
  {
    out << "Level " << p->get_level() << "\n";
    out << "Number of children: " << p->get_child_num() << "\n";

    out << "Node data:\n" << p->get_data_reference();
    out << "Split mode: " << p->get_split_mode() << "\n";

    if (p->Parent() != nullptr)
      {
        out << "Parent node data: " << p->Parent()->get_data_reference()
            << "\n";
      }
    else
      {
        out << "Parent node data: none\n";
      }

    out << "------------------" << std::endl;
  }

  /**
   * Calculate the depth of a BinaryTree using recursion.
   */
  template <typename T>
  unsigned int
  calc_depth(const BinaryTreeNode<T> *p)
  {
    if (p == nullptr)
      {
        return 0;
      }
    else
      {
        return std::max(calc_depth(p->Left()), calc_depth(p->Right())) + 1;
      }
  }

  /**
   * Calculate the depth of a TreeNode using recursion.
   */
  template <typename T, std::size_t N>
  unsigned int
  calc_depth(TreeNode<T, N> *p)
  {
    if (p == nullptr)
      {
        return 0;
      }
    else
      {
        // Array storing the depth of all subtrees.
        std::array<unsigned int, N> subtree_depths;

        for (std::size_t i = 0; i < N; i++)
          {
            subtree_depths.at(i) = calc_depth(p->get_child_pointer(i));
          }

        return (*std::max_element(subtree_depths.cbegin(),
                                  subtree_depths.cend())) +
               1;
      }
  }

  /**
   * Pre-order traverse of a binary tree.
   */
  template <typename T>
  void
  Preorder(BinaryTreeNode<T>                       *p,
           std::function<void(BinaryTreeNode<T> *)> operate)
  {
    if (p != nullptr)
      {
        operate(p);
        Preorder(p->Left(), operate);
        Preorder(p->Right(), operate);
      }
  }

  /**
   * Pre-order traverse of a binary tree (const version).
   */
  template <typename T>
  void
  Preorder(const BinaryTreeNode<T>                       *p,
           std::function<void(const BinaryTreeNode<T> *)> operate)
  {
    if (p != nullptr)
      {
        operate(p);
        Preorder(p->Left(), operate);
        Preorder(p->Right(), operate);
      }
  }

  /**
   * Pre-order traverse of a tree.
   */
  template <typename T, std::size_t N>
  void
  Preorder(TreeNode<T, N> *p, std::function<void(TreeNode<T, N> *)> operate)
  {
    if (p != nullptr)
      {
        operate(p);

        /**
         * Recursively call the function itself on each child.
         */
        for (std::size_t i = 0; i < N; i++)
          {
            Preorder(p->get_child_pointer(i), operate);
          }
      }
  }

  /**
   * Pre-order traverse of a tree (const version).
   */
  template <typename T, std::size_t N>
  void
  Preorder(const TreeNode<T, N>                       *p,
           std::function<void(const TreeNode<T, N> *)> operate)
  {
    if (p != nullptr)
      {
        operate(p);

        /**
         * Recursively call the operate function on each child.
         */
        for (std::size_t i = 0; i < N; i++)
          {
            Preorder(p->get_child_pointer(i), operate);
          }
      }
  }

  /**
   * In-order traverse of a binary tree.
   */
  template <typename T>
  void
  Inorder(BinaryTreeNode<T>                       *p,
          std::function<void(BinaryTreeNode<T> *)> operate)
  {
    if (p != nullptr)
      {
        Inorder(p->Left(), operate);
        operate(p);
        Inorder(p->Right(), operate);
      }
  }

  /**
   * In-order traverse of a binary tree (const version).
   */
  template <typename T>
  void
  Inorder(const BinaryTreeNode<T>                       *p,
          std::function<void(const BinaryTreeNode<T> *)> operate)
  {
    if (p != nullptr)
      {
        Inorder(p->Left(), operate);
        operate(p);
        Inorder(p->Right(), operate);
      }
  }

  /**
   * Post-order traverse of a binary tree.
   */
  template <typename T>
  void
  Postorder(BinaryTreeNode<T>                       *p,
            std::function<void(BinaryTreeNode<T> *)> operate)
  {
    if (p != nullptr)
      {
        Postorder(p->Left(), operate);
        Postorder(p->Right(), operate);
        operate(p);
      }
  }

  /**
   * Post-order traverse of a binary tree (const version).
   */
  template <typename T>
  void
  Postorder(const BinaryTreeNode<T>                       *p,
            std::function<void(const BinaryTreeNode<T> *)> operate)
  {
    if (p != nullptr)
      {
        Postorder(p->Left(), operate);
        Postorder(p->Right(), operate);
        operate(p);
      }
  }

  /**
   * Post-order traverse of a tree.
   */
  template <typename T, std::size_t N>
  void
  Postorder(TreeNode<T, N> *p, std::function<void(TreeNode<T, N> *)> operate)
  {
    if (p != nullptr)
      {
        /**
         * Recursively call the operate function on each child.
         */
        for (std::size_t i = 0; i < N; i++)
          {
            Postorder(p->get_child_pointer(i), operate);
          }

        operate(p);
      }
  }

  /**
   * Post-order traverse of a tree (const version).
   */
  template <typename T, std::size_t N>
  void
  Postorder(const TreeNode<T, N>                       *p,
            std::function<void(const TreeNode<T, N> *)> operate)
  {
    if (p != nullptr)
      {
        /**
         * Recursively call the operate function on each child.
         */
        for (std::size_t i = 0; i < N; i++)
          {
            Postorder(p->get_child_pointer(i), operate);
          }

        operate(p);
      }
  }


  template <typename T>
  BinaryTreeNode<T> *
  CopyTree(const BinaryTreeNode<T> *p)
  {
    BinaryTreeNode<T> *current_node = nullptr;

    if (p != nullptr)
      {
        /**
         * Copy the current node to a new node without setting children, parent
         * and child_num at the moment.
         */
        current_node = CreateTreeNode(p->get_data_reference(), p->get_level());

        BinaryTreeNode<T> *left_child_node  = CopyTree(p->Left());
        BinaryTreeNode<T> *right_child_node = CopyTree(p->Right());

        if (left_child_node != nullptr)
          {
            /**
             * Associate the newly created left child node with the current
             * node.
             */
            left_child_node->Parent(current_node);
            current_node->Left(left_child_node);

            current_node->increase_child_num();
          }

        if (right_child_node != nullptr)
          {
            /**
             * Associate the newly created right child node with the current
             * node.
             */
            right_child_node->Parent(current_node);
            current_node->Right(right_child_node);

            current_node->increase_child_num();
          }

        return current_node;
      }
    else
      {
        return current_node;
      }
  }


  /**
   * Perform deep copy of a tree.
   * @param p
   * @return
   */
  template <typename T, std::size_t N>
  TreeNode<T, N> *
  CopyTree(const TreeNode<T, N> *p)
  {
    TreeNode<T, N> *current_node = nullptr;

    if (p != nullptr)
      {
        // Initialize the four null child pointers.
        std::array<TreeNode<T, N> *, N> empty_child_pointers;
        for (unsigned int i = 0; i < N; i++)
          {
            empty_child_pointers[i] = nullptr;
          }

        /**
         * Create and copy the current node.
         */
        current_node = CreateTreeNode(p->get_data_reference(),
                                      p->get_level(),
                                      empty_child_pointers,
                                      static_cast<TreeNode<T, N> *>(nullptr),
                                      p->get_split_mode());

        for (unsigned int i = 0; i < N; i++)
          {
            TreeNode<T, N> *child_node = CopyTree(p->get_child_pointer(i));

            if (child_node != nullptr)
              {
                child_node->Parent(current_node);
                current_node->set_child_pointer(current_node->get_child_num(),
                                                child_node);

                current_node->increase_child_num();
              }
          }

        return current_node;
      }
    else
      {
        return current_node;
      }
  }


  /**
   * Construct the leaf set of a binary tree.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The leaf set should be cleared before calling this function.</dd>
   * </dl>
   *
   * @param p
   * @param leaf_set
   */
  template <typename T>
  void
  GetTreeLeaves(const BinaryTreeNode<T>          *p,
                std::vector<BinaryTreeNode<T> *> &leaf_set)
  {
    if (p->get_child_num() == 0)
      {
        /**
         * If the current node has no children, append it to the leaf set.
         */
        if (p != nullptr)
          {
            leaf_set.push_back(const_cast<BinaryTreeNode<T> *>(p));
          }
        else
          {
            Assert(false, dealii::ExcInternalError());
          }
      }
    else
      {
        /**
         * If the current node has children, recursively collect leaves from its
         * left and right children.
         */
        if (p->Left() != nullptr)
          {
            GetTreeLeaves(p->Left(), leaf_set);
          }

        if (p->Right() != nullptr)
          {
            GetTreeLeaves(p->Right(), leaf_set);
          }
      }
  }


  /**
   * Construct the leaf set of a general tree.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The leaf set should be cleared before calling this function.</dd>
   * </dl>
   *
   * @param p
   * @param leaf_set
   */
  template <typename T, std::size_t N>
  void
  GetTreeLeaves(const TreeNode<T, N>          *p,
                std::vector<TreeNode<T, N> *> &leaf_set)
  {
    if (p->get_child_num() == 0)
      {
        /**
         * If the current node has no children, append it to the leaf set.
         */
        if (p != nullptr)
          {
            leaf_set.push_back(const_cast<TreeNode<T, N> *>(p));
          }
        else
          {
            Assert(false, dealii::ExcInternalError());
          }
      }
    else
      {
        /**
         * If the current node has children, recursively collect leaves from
         * each of its children.
         */
        for (std::size_t i = 0; i < N; i++)
          {
            if (p->get_child_pointer(i) != nullptr)
              {
                GetTreeLeaves(p->get_child_pointer(i), leaf_set);
              }
          }
      }
  }


  /**
   * Construct the set of nodes at the specified level of a binary tree.
   *
   * @level starts from zero.
   *
   * @tparam T
   * @param p
   * @param level
   * @param node_set
   */
  template <typename T>
  void
  GetTreeNodesAtLevel(const BinaryTreeNode<T>          *p,
                      const unsigned int                level,
                      std::vector<BinaryTreeNode<T> *> &node_set)
  {
    if (p->get_level() == level)
      {
        if (p != nullptr)
          {
            node_set.push_back(const_cast<BinaryTreeNode<T> *>(p));
          }
        else
          {
            Assert(false, dealii::ExcInternalError());
          }
      }
    else
      {
        Assert(p->get_level() < level, dealii::ExcInternalError());

        /**
         * If the current node has children, recursively collect nodes at the
         * specified level from its left and right children.
         */
        if (p->Left() != nullptr)
          {
            GetTreeNodesAtLevel(p->Left(), level, node_set);
          }

        if (p->Right() != nullptr)
          {
            GetTreeNodesAtLevel(p->Right(), level, node_set);
          }
      }
  }


  /**
   * Construct the set of nodes at the specified level of a general tree.
   *
   * @level starts from zero.
   *
   * @tparam T
   * @tparam N
   * @param p
   * @param level
   * @param node_set
   */
  template <typename T, std::size_t N>
  void
  GetTreeNodesAtLevel(const TreeNode<T, N>          *p,
                      const unsigned int             level,
                      std::vector<TreeNode<T, N> *> &node_set)
  {
    if (p->get_level() == level)
      {
        if (p != nullptr)
          {
            node_set.push_back(const_cast<TreeNode<T, N> *>(p));
          }
        else
          {
            Assert(false, dealii::ExcInternalError());
          }
      }
    else
      {
        /**
         * If the current node has children, recursively collect nodes at the
         * specified level from each of its children.
         */
        for (std::size_t i = 0; i < N; i++)
          {
            if (p->get_child_pointer(i) != nullptr)
              {
                GetTreeNodesAtLevel(p->get_child_pointer(i), level, node_set);
              }
          }
      }
  }


  /**
   * Print a binary tree recursively by starting from a node.
   */
  template <typename T>
  void
  PrintTree(std::ostream &out, const BinaryTreeNode<T> *p)
  {
    if (p != nullptr)
      {
        Preorder<T>(p,
                    std::bind(&PrintTreeNode<T>,
                              std::ref(out),
                              std::placeholders::_1));
      }
  }

  /**
   * Print a tree recursively by starting from a node.
   */
  template <typename T, std::size_t N>
  void
  PrintTree(std::ostream &out, const TreeNode<T, N> *p)
  {
    if (p != nullptr)
      {
        Preorder<T, N>(p,
                       std::bind(&PrintTreeNode<T, N>,
                                 std::ref(out),
                                 std::placeholders::_1));
      }
  }


  /**
   * Count the total number of the binary tree nodes by recursion.
   */
  template <typename T>
  unsigned int
  CountTreeNodes(const BinaryTreeNode<T> *p)
  {
    unsigned int node_num = 0;

    if (p != nullptr)
      {
        node_num++;
        node_num += CountTreeNodes(p->Left());
        node_num += CountTreeNodes(p->Right());

        return node_num;
      }
    else
      {
        return 0;
      }
  }


  /**
   * Count the total number of the tree nodes by recursion.
   */
  template <typename T, std::size_t N>
  unsigned int
  CountTreeNodes(const TreeNode<T, N> *p)
  {
    unsigned int node_num = 0;

    if (p != nullptr)
      {
        node_num++;
        for (unsigned int i = 0; i < N; i++)
          {
            node_num += CountTreeNodes(p->get_child_pointer(i));
          }

        return node_num;
      }
    else
      {
        return 0;
      }
  }


  /**
   * Use the Delete a whole binary tree using post-order traversal.
   */
  template <typename T>
  void
  DeleteTree(const BinaryTreeNode<T> *p)
  {
    Postorder<T>(p, DeleteTreeNode<T>);
  }

  /**
   * Use the Delete a whole tree using post-order traversal.
   */
  template <typename T, std::size_t N>
  void
  DeleteTree(const TreeNode<T, N> *p)
  {
    Postorder<T>(p, DeleteTreeNode<T, N>);
  }
} // namespace HierBEM

/**
 * @}
 */

#endif /* TREE_H_ */
