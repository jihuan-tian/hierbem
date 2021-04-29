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

#include <array>
#include <exception>
#include <functional>
#include <iostream>
#include <new>

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
  BinaryTreeNode(const T &       data,
                 unsigned int    level,
                 BinaryTreeNode *left  = nullptr,
                 BinaryTreeNode *right = nullptr);

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
   * Set the left child node pointer.
   *
   * N.B. The const pointer type in the argument will be converted to non-const
   * pointer type. In this way, even a const node can be added to the tree.
   */
  void
  Left(const BinaryTreeNode *node_pointer);

  /**
   * Set the right child node pointer.
   *
   * N.B. The const pointer type in the argument will be converted to non-const
   * pointer type. In this way, even a const node can be added to the tree.
   */
  void
  Right(const BinaryTreeNode *node_pointer);

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
   * Get the level of the node in the tree.
   */
  unsigned int
  get_level() const;

private:
  T            data;
  unsigned int level;

  BinaryTreeNode *left;
  BinaryTreeNode *right;
};

template <typename T>
BinaryTreeNode<T>::BinaryTreeNode()
  : data()
  , level(0)
  , left(nullptr)
  , right(nullptr)
{}

template <typename T>
BinaryTreeNode<T>::BinaryTreeNode(const T &       data,
                                  unsigned int    level,
                                  BinaryTreeNode *left,
                                  BinaryTreeNode *right)
  : data(data)
  , level(level)
  , left(left)
  , right(right)
{}

template <typename T>
BinaryTreeNode<T>::BinaryTreeNode(const BinaryTreeNode &node)
  : data(node.data)
  , level(node.level)
  , left(node.left)
  , right(node.right)
{}

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
  Assert((index >= 0) && (index < 2), dealii::ExcIndexRange(index, 0, 2));

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

/**
 * \brief Class for general tree node.
 *
 * This general tree node can contains any but \em fixed number of children. \p
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
   */
  TreeNode(const T &                        data,
           unsigned int                     level,
           const std::array<TreeNode *, N> &children);

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
   * N.B. The const pointer type in the argument will be converted to non-const
   * pointer type. In this way, even a const node can be added to the tree.
   * @param index Index of the child
   * @param pointer New pointer value
   */
  void
  set_child_pointer(std::size_t index, const TreeNode *pointer);

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
   * Get the level of the node in the tree.
   */
  unsigned int
  get_level() const;

private:
  T            data;
  unsigned int level;

  std::array<TreeNode *, N> children;
};

template <typename T, std::size_t N>
TreeNode<T, N>::TreeNode()
  : data()
  , level(0)
{
  children.fill(nullptr);
}

template <typename T, std::size_t N>
TreeNode<T, N>::TreeNode(const T &data)
  : data(data)
  , level(0)
{
  children.fill(nullptr);
}

template <typename T, std::size_t N>
TreeNode<T, N>::TreeNode(const T &                        data,
                         unsigned int                     level,
                         const std::array<TreeNode *, N> &children)
  : data(data)
  , level(level)
  , children(children)
{}

template <typename T, std::size_t N>
TreeNode<T, N>::TreeNode(const TreeNode &node)
  : data(node.data)
  , level(node.level)
  , children(node.children)
{}

template <typename T, std::size_t N>
TreeNode<T, N> *
TreeNode<T, N>::get_child_pointer(std::size_t index) const
{
  return children.at(index);
}

template <typename T, std::size_t N>
void
TreeNode<T, N>::set_child_pointer(std::size_t           index,
                                  const TreeNode<T, N> *pointer)
{
  children.at(index) = const_cast<TreeNode *>(pointer);
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
unsigned int
TreeNode<T, N>::get_level() const
{
  return level;
}

/**
 * Create a new binary tree node from the provided data.
 */
template <typename T>
BinaryTreeNode<T> *
CreateTreeNode(const T &          data,
               unsigned int       level = 0,
               BinaryTreeNode<T> *left  = nullptr,
               BinaryTreeNode<T> *right = nullptr)
{
  BinaryTreeNode<T> *p = nullptr;
  p                    = new BinaryTreeNode<T>(data, level, left, right);

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
CreateTreeNode(const T &                              data,
               unsigned int                           level,
               const std::array<TreeNode<T, N> *, N> &children)
{
  TreeNode<T, N> *p = nullptr;
  p                 = new TreeNode<T, N>(data, level, children);

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
  out << *(p->get_data_pointer()) << "\n------------------\n" << std::endl;
}

/**
 * Print the data contained in a tree node.
 */
template <typename T, std::size_t N>
void
PrintTreeNode(std::ostream &out, const TreeNode<T, N> *p)
{
  out << "Level " << p->get_level() << "\n";
  out << *(p->get_data_pointer()) << "\n------------------\n" << std::endl;
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
Preorder(BinaryTreeNode<T> *p, std::function<void(BinaryTreeNode<T> *)> operate)
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
Preorder(const BinaryTreeNode<T> *                      p,
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
Preorder(const TreeNode<T, N> *                      p,
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
Inorder(BinaryTreeNode<T> *p, std::function<void(BinaryTreeNode<T> *)> operate)
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
Inorder(const BinaryTreeNode<T> *                      p,
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
Postorder(BinaryTreeNode<T> *                      p,
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
Postorder(const BinaryTreeNode<T> *                      p,
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
Postorder(const TreeNode<T, N> *                      p,
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

/**
 * Print a binary tree recursively by starting from a node.
 */
template <typename T>
void
PrintTree(std::ostream &out, const BinaryTreeNode<T> *p)
{
  if (p != nullptr)
    {
      Preorder<T>(
        p, std::bind(&PrintTreeNode<T>, std::ref(out), std::placeholders::_1));
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

/**
 * @}
 */

#endif /* TREE_H_ */
