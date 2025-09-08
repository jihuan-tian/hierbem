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
 * \file quad-tree-get-leaves.cc
 * \brief Verify extraction of the leaves of a tree.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-21
 */

#include <iostream>
#include <vector>

#include "cluster_tree/tree.h"
#include "utilities/debug_tools.h"

using namespace HierBEM;

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
  parent->set_split_mode(CrossSplitMode);
  auto *node2 = new TreeNode<int, 4>(2);
  auto *node3 = new TreeNode<int, 4>(3);
  node3->set_split_mode(CrossSplitMode);
  auto *node4 = new TreeNode<int, 4>(4);
  auto *node5 = new TreeNode<int, 4>(5);
  auto *node6 = new TreeNode<int, 4>(6);
  auto *node7 = new TreeNode<int, 4>(7);
  auto *node8 = new TreeNode<int, 4>(8);
  auto *node9 = new TreeNode<int, 4>(9);

  std::array<TreeNode<int, 4> *, 4> children_for_10{
    {node3, node4, node5, node6}};

  for (std::size_t i = 0; i < 4; i++)
    {
      parent->set_child_pointer(i, children_for_10.at(i));
      children_for_10.at(i)->Parent(parent);

      children_for_10.at(i)->set_level(parent->get_level() + 1);
      parent->increase_child_num();
    }

  std::array<TreeNode<int, 4> *, 4> children_for_3{
    {node2, node8, node7, node9}};

  for (std::size_t i = 0; i < 4; i++)
    {
      node3->set_child_pointer(i, children_for_3.at(i));
      children_for_3.at(i)->Parent(node3);

      children_for_3.at(i)->set_level(node3->get_level() + 1);
      node3->increase_child_num();
    }

  return parent;
}

int
main()
{
  TreeNode<int, 4> *tree = MakeIntExampleTree();

  std::vector<TreeNode<int, 4> *> leaf_set;
  GetTreeLeaves(tree, leaf_set);

  print_vector_of_tree_node_pointer_values(std::cout, leaf_set, "\n");

  return 0;
}
