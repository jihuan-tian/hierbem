#include <functional>
#include <iostream>

#include "cluster_tree/tree.h"

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
  BinaryTreeNode<int> *root = MakeIntExampleTree();

  // The answer should be: [10, 3, 2, 4, 8, 6, 5, 7, 9].
  std::cout << "Pre-order traverse of the tree:\n";
  Preorder<int>(root,
                std::bind(&PrintTreeNode<int>,
                          std::ref(std::cout),
                          std::placeholders::_1));

  // The answer should be: [2, 3, 8, 4, 6, 10, 7, 5, 9].
  std::cout << "In-order traverse of the tree:\n";
  Inorder<int>(root,
               std::bind(&PrintTreeNode<int>,
                         std::ref(std::cout),
                         std::placeholders::_1));

  // The answer should be: [2, 8, 6, 4, 3, 7, 9, 5, 10].
  std::cout << "Post-order traverse of the tree:\n";
  Postorder<int>(root,
                 std::bind(&PrintTreeNode<int>,
                           std::ref(std::cout),
                           std::placeholders::_1));

  std::cout << "Print the tree using PrintTree:\n";
  PrintTree(std::cout, root);

  // Delete the tree.
  DeleteTree(root);

  return 0;
}
