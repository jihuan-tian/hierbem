// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file quad-tree-memory-consumption.cc
 * @brief Compute the memory consumption of a quad tree.
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
 *           10
 *       /  |  |  \
 *      3   4  5   6
 *   / | |  \
 *  2  8  7  9
 *
 * The function returns the pointer to the root node of the tree.
 */
TreeNode<int, 4> *
MakeIntExampleTree()
{
  auto *parent = new TreeNode<int, 4>(10);
  cout << parent->memory_consumption() << "\n";
  parent->set_split_mode(CrossSplitMode);
  auto *node2 = new TreeNode<int, 4>(2);
  cout << node2->memory_consumption() << "\n";
  auto *node3 = new TreeNode<int, 4>(3);
  cout << node3->memory_consumption() << "\n";
  node3->set_split_mode(CrossSplitMode);
  auto *node4 = new TreeNode<int, 4>(4);
  cout << node4->memory_consumption() << "\n";
  auto *node5 = new TreeNode<int, 4>(5);
  cout << node5->memory_consumption() << "\n";
  auto *node6 = new TreeNode<int, 4>(6);
  cout << node6->memory_consumption() << "\n";
  auto *node7 = new TreeNode<int, 4>(7);
  cout << node7->memory_consumption() << "\n";
  auto *node8 = new TreeNode<int, 4>(8);
  cout << node8->memory_consumption() << "\n";
  auto *node9 = new TreeNode<int, 4>(9);
  cout << node9->memory_consumption() << "\n";

  std::array<TreeNode<int, 4> *, 4> children_for_10{
    {node3, node4, node5, node6}};

  for (std::size_t i = 0; i < 4; i++)
    {
      parent->set_child_pointer(i, children_for_10.at(i));
    }

  std::array<TreeNode<int, 4> *, 4> children_for_3{
    {node2, node8, node7, node9}};

  for (std::size_t i = 0; i < 4; i++)
    {
      node3->set_child_pointer(i, children_for_3.at(i));
    }

  return parent;
}


int
main()
{
  TreeNode<int, 4> *root = MakeIntExampleTree();

  std::size_t memory_size = 0;
  Preorder_for_memory_consumption(root, memory_size);

  cout << "sizeof(TreeNodeSplitMode): " << sizeof(TreeNodeSplitMode) << "\n";
  cout << "Memory consumption: " << memory_size << endl;

  // Delete the tree.
  DeleteTree(root);

  return 0;
}
