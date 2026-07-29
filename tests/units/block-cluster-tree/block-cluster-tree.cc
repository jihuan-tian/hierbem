// Copyright (C) 2021-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file block-cluster-tree.cc
 * \brief This files verifies the admissible block cluster partition for a
 * mesh.
 *
 * \ingroup test_cases hierarchical_matrices
 * \date 2021-04-28
 * \author Jihuan Tian
 */

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#include "cluster_tree/block_cluster_tree.h"
#include "cluster_tree/cluster_tree.h"
#include "dofs/dof_tools_ext.h"
#include "hbem_cpp_validate.h"
#include "utilities/debug_tools.h"

using namespace HierBEM;
using namespace dealii;
using namespace Catch::Matchers;

TEST_CASE("Construct block cluster tree", "[hmatrix]")
{
  // Generate the 3x3 grid in a 2D square.
  const unsigned int           spacedim = 3;
  const unsigned int           dim      = 2;
  Triangulation<dim, spacedim> triangulation;

  std::vector<unsigned int> n_subdivisions{10, 10};
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            n_subdivisions,
                                            Point<dim>(0, 0),
                                            Point<dim>(1, 1));

  // Save the mesh to a file for visualization.
  GridOut       grid_out;
  std::ofstream mesh_file("square.msh");
  grid_out.write_msh(triangulation, mesh_file);

  // Create the Lagrangian finite element.
  const unsigned int  fe_order = 1;
  FE_Q<dim, spacedim> fe(fe_order);

  // Create a DoFHandler, which is associated with the triangulation and
  // distributed with the finite element.
  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Create the mapping object, which is required in generating the map from
  // DoF indices to support points.
  const MappingQ<dim, spacedim> mapping(1);

  // Generate a list of all DoF indices.
  std::vector<types::global_dof_index> dof_indices(dof_handler.n_dofs());
  types::global_dof_index              counter = 0;
  for (auto &dof_index : dof_indices)
    {
      dof_index = counter;
      counter++;
    }

  // Get the spatial coordinates of the support points associated with DoF
  // indices.
  std::vector<Point<spacedim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);


  // Calculate the DoF support set diameter at each support point.
  std::vector<double> dof_support_set_diameters(dof_handler.n_dofs(), 0);
  DoFToolsExt::map_dofs_to_support_set_diameters(dof_handler,
                                                 dof_support_set_diameters);

  SECTION("build tree in serial")
  {
    std::ofstream ofs("block-cluster-tree-serial.log");

    // Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
    // indices.
    const unsigned int    n_min = 4;
    ClusterTree<spacedim> TI(dof_indices,
                             all_support_points,
                             dof_support_set_diameters,
                             n_min);

    // Partition the cluster tree.
    TI.partition(all_support_points, dof_support_set_diameters);

    unsigned int dofs_in_leaf_set = 0;
    for (auto node : TI.get_leaf_set())
      dofs_in_leaf_set += node->get_data_reference().get_cardinality();

    ofs << "Total number of DoFs: " << all_support_points.size() << "\n";
    ofs << "Total number of DoFs in cluster tree leaf set: " << dofs_in_leaf_set
        << "\n";

    // Print the cluster tree.
    ofs << "=== Cluster tree TI ===\n";
    ofs << TI << std::endl;

    // Create the block cluster tree.
    const double               eta = 10;
    BlockClusterTree<spacedim> block_cluster_tree(TI, TI, eta, n_min);

    // Perform admissible partition on the block cluster tree.
    block_cluster_tree.partition();

    // Print the block cluster tree, even though there is only a root node.
    ofs << "=== Block cluster tree ===\n";
    ofs << block_cluster_tree << "\n";
    ofs << "Memory consumption of all block clusters: "
        << block_cluster_tree.memory_consumption_of_all_block_clusters()
        << "\n";
    ofs << "Memory consumption: " << block_cluster_tree.memory_consumption()
        << std::endl;

    ofs.close();

    compare_two_files(SOURCE_DIR "/reference.output",
                      "block-cluster-tree-serial.log");
  }

  SECTION("build tree in parallel")
  {
    std::ofstream ofs("block-cluster-tree-parallel.log");

    // Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
    // indices.
    const unsigned int    n_min            = 4;
    const unsigned int    cutoff_level_ct  = 2;
    const unsigned int    cutoff_level_bct = 2;
    ClusterTree<spacedim> TI(dof_indices,
                             all_support_points,
                             dof_support_set_diameters,
                             n_min);

    // Partition the cluster tree.
    TI.partition(all_support_points,
                 dof_support_set_diameters,
                 cutoff_level_ct);

    unsigned int dofs_in_leaf_set = 0;
    for (auto node : TI.get_leaf_set())
      dofs_in_leaf_set += node->get_data_reference().get_cardinality();

    ofs << "Total number of DoFs: " << all_support_points.size() << "\n";
    ofs << "Total number of DoFs in cluster tree leaf set: " << dofs_in_leaf_set
        << "\n";

    // Print the cluster tree.
    ofs << "=== Cluster tree TI ===\n";
    ofs << TI << std::endl;

    // Create the block cluster tree.
    const double               eta = 10;
    BlockClusterTree<spacedim> block_cluster_tree(TI, TI, eta, n_min);

    // Perform admissible partition on the block cluster tree.
    block_cluster_tree.partition(cutoff_level_bct);

    // Print the block cluster tree, even though there is only a root node.
    ofs << "=== Block cluster tree ===\n";
    ofs << block_cluster_tree << "\n";
    ofs << "Memory consumption of all block clusters: "
        << block_cluster_tree.memory_consumption_of_all_block_clusters()
        << "\n";
    ofs << "Memory consumption: " << block_cluster_tree.memory_consumption()
        << std::endl;

    ofs.close();

    compare_two_files(SOURCE_DIR "/reference.output",
                      "block-cluster-tree-serial.log");
  }
}
