// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file binary-tree-memory-consumption.cc
 * @brief Compute the memory consumption of a binary tree.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-02-19
 */

#include <deal.II/base/memory_consumption.h>

#include <iostream>

#include "cluster_tree/tree.h"

using namespace HierBEM;

using namespace std;

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
  left_child = CreateTreeNode<int>(8, 3);
  cout << left_child->memory_consumption() << "\n";
  right_child = CreateTreeNode<int>(6, 3);
  cout << right_child->memory_consumption() << "\n";
  parent = CreateTreeNode<int>(4, 2, left_child, right_child);
  cout << parent->memory_consumption() << "\n";
  left_child->Parent(parent);
  right_child->Parent(parent);

  right_child = parent;
  left_child  = CreateTreeNode<int>(2, 2);
  cout << left_child->memory_consumption() << "\n";
  parent1 = CreateTreeNode<int>(3, 1, left_child, right_child);
  cout << parent1->memory_consumption() << "\n";
  left_child->Parent(parent1);
  right_child->Parent(parent1);

  // Create the right subtree of the root node.
  left_child = CreateTreeNode<int>(7, 2);
  cout << left_child->memory_consumption() << "\n";
  right_child = CreateTreeNode<int>(9, 2);
  cout << right_child->memory_consumption() << "\n";
  parent2 = CreateTreeNode<int>(5, 1, left_child, right_child);
  cout << parent2->memory_consumption() << "\n";
  left_child->Parent(parent2);
  right_child->Parent(parent2);

  parent = CreateTreeNode<int>(10, 0, parent1, parent2);
  cout << parent->memory_consumption() << "\n";
  parent1->Parent(parent);
  parent2->Parent(parent);

  return parent;
}

int
main()
{
  BinaryTreeNode<int> *root = MakeIntExampleTree();

  std::size_t memory_size = 0;

  Preorder_for_memory_consumption(root, memory_size);

  cout << "Memory consumption: " << memory_size << endl;

  // Delete the tree.
  DeleteTree(root);

  return 0;
}
