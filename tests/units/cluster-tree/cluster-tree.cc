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
 * @file cluster-tree.cc
 * This file verifies the ClusterTree class.
 *
 * @author Jihuan Tian
 */

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <catch2/catch_all.hpp>
#include <cluster_tree/simple_bounding_box.h>

#include <fstream>
#include <iostream>

#include "cluster_tree/cluster_tree.h"
#include "hbem_cpp_validate.h"
#include "utilities/debug_tools.h"

using namespace HierBEM;
using namespace dealii;
using namespace Catch::Matchers;

TEST_CASE("Construct cluster tree", "[hmatrix]")
{
  // Generate the grid for a 3D sphere.
  const unsigned int      dim = 3;
  Triangulation<dim, dim> triangulation;
  // N.B. Use type cast for triangulation to suppress Eclipse editor error
  // prompt.
  GridGenerator::hyper_ball((Triangulation<dim> &)triangulation,
                            Point<3>(0., 0., 0.),
                            2.0,
                            true);
  triangulation.refine_global(3);

  // Create a Lagrangian finite element.
  FE_Q<dim, dim> fe(1);

  // Create a DoFHandler, which is associated with the triangulation and
  // distributed with the finite element.
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  // Create a mapping object, which is required in generating the map from
  // DoF indices to support points.
  const MappingQ<dim, dim> mapping(1);

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
  std::vector<Point<dim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);

  // Initialize a cluster tree for all the DoF indices.
  const unsigned int n_min        = 4;
  const unsigned int cutoff_level = 8;

  {
    std::ofstream    ofs("cluster-tree-parallel.log");
    Timer            timer;
    ClusterTree<dim> cluster_tree(dof_indices, all_support_points, n_min);
    timer.stop();
    print_wall_time(std::cout, timer, "create root node");

    // Partition the cluster tree.
    timer.start();
    cluster_tree.partition(all_support_points, cutoff_level);
    timer.stop();
    print_wall_time(std::cout, timer, "partition cluster tree in parallel");

    // Print the coordinates of all support points.
    ofs << "=== Support point coordinates ===\n";
    for (auto &point : all_support_points)
      {
        ofs << point << "\n";
      }

    // Print the whole cluster tree.
    ofs << "=== Cluster tree ===\n";
    ofs << cluster_tree << std::endl;

    // Compute the memory consumption.
    ofs << "Memory consumption of all clusters: "
        << cluster_tree.memory_consumption_of_all_clusters() << "\n";
    ofs << "Memory consumption: " << cluster_tree.memory_consumption()
        << std::endl;
    ofs.close();

    // compare_two_files("cluster-tree.log", SOURCE_DIR "/reference.output");
  }

  {
    std::ofstream    ofs("cluster-tree-serial.log");
    ClusterTree<dim> cluster_tree(dof_indices, all_support_points, n_min);

    // Partition the cluster tree.
    Timer timer;
    cluster_tree.partition(all_support_points);
    timer.stop();
    print_wall_time(std::cout, timer, "partition cluster tree in serial");

    // Print the coordinates of all support points.
    ofs << "=== Support point coordinates ===\n";
    for (auto &point : all_support_points)
      {
        ofs << point << "\n";
      }

    // Print the whole cluster tree.
    ofs << "=== Cluster tree ===\n";
    ofs << cluster_tree << std::endl;

    // Compute the memory consumption.
    ofs << "Memory consumption of all clusters: "
        << cluster_tree.memory_consumption_of_all_clusters() << "\n";
    ofs << "Memory consumption: " << cluster_tree.memory_consumption()
        << std::endl;
    ofs.close();

    // Export the cluster tree as a directional graph.
    std::ofstream graph("cluster-tree.puml");
    cluster_tree.print_tree_info_as_dot(graph);
    graph.close();

    // compare_two_files("cluster-tree.log", SOURCE_DIR "/reference.output");
  }

  compare_two_files("cluster-tree-serial.log", "cluster-tree-parallel.log");
}
