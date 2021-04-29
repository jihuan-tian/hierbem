#include <functional>
#include <iostream>

#include "tree.h"

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
  BinaryTreeNode<int> *parent, *left_child, *right_child;

  // Create the left subtree of the root node.
  left_child  = CreateTreeNode<int>(8);
  right_child = CreateTreeNode<int>(6);
  parent      = CreateTreeNode<int>(4, 0, left_child, right_child);
  right_child = parent;
  left_child  = CreateTreeNode<int>(2);
  parent      = CreateTreeNode<int>(3, 0, left_child, right_child);


  // Create the right subtree of the root node.
  left_child  = CreateTreeNode<int>(7);
  right_child = CreateTreeNode<int>(9);
  right_child = CreateTreeNode<int>(5, 0, left_child, right_child);
  left_child  = parent;
  parent      = CreateTreeNode<int>(10, 0, left_child, right_child);

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
