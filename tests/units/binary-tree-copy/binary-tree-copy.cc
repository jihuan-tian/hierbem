/**
 * \file binary-tree-copy.cc
 * \brief Verify the copy constructor of a binary tree.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-20
 */

#include <functional>
#include <iostream>

#include "tree.h"

using namespace HierBEM;

/**
 * Create an example tree containing integers as below. The tree will be
 * constructed in a bottom-up approach.
 *
 *          10
 *         /  \
 *       3      5
 *      / \    / \
 *     2   4  7   9
 *        / \
 *       8   6
 *
 * The function returns the pointer to the root node of the tree.
 */
BinaryTreeNode<int> *
MakeIntExampleTree()
{
  BinaryTreeNode<int> *parent, *parent1, *parent2, *left_child, *right_child;

  // Create the left subtree of the root node.
  left_child  = CreateTreeNode<int>(8, 3);
  right_child = CreateTreeNode<int>(6, 3);
  parent      = CreateTreeNode<int>(4, 2, left_child, right_child);
  left_child->Parent(parent);
  right_child->Parent(parent);

  right_child = parent;
  left_child  = CreateTreeNode<int>(2, 2);
  parent1     = CreateTreeNode<int>(3, 1, left_child, right_child);
  left_child->Parent(parent1);
  right_child->Parent(parent1);

  // Create the right subtree of the root node.
  left_child  = CreateTreeNode<int>(7, 2);
  right_child = CreateTreeNode<int>(9, 2);
  parent2     = CreateTreeNode<int>(5, 1, left_child, right_child);
  left_child->Parent(parent2);
  right_child->Parent(parent2);

  parent = CreateTreeNode<int>(10, 0, parent1, parent2);
  parent1->Parent(parent);
  parent2->Parent(parent);

  return parent;
}

int
main()
{
  BinaryTreeNode<int> *tree_orig = MakeIntExampleTree();
  BinaryTreeNode<int> *tree_copy = CopyTree(tree_orig);

  // The answer should be: [10, 3, 2, 4, 8, 6, 5, 7, 9].
  std::cout << "Original tree:\n";
  Preorder<int>(tree_orig,
                std::bind(&PrintTreeNode<int>,
                          std::ref(std::cout),
                          std::placeholders::_1));

  DeleteTree(tree_orig);

  std::cout << "Copied tree:\n";
  Preorder<int>(tree_copy,
                std::bind(&PrintTreeNode<int>,
                          std::ref(std::cout),
                          std::placeholders::_1));

  DeleteTree(tree_copy);

  return 0;
}
