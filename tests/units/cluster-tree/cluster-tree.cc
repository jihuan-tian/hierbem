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
 * \file cluster-tree.cc
 * This file verifies the ClusterTree class.
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <cluster_tree/simple_bounding_box.h>

#include <fstream>

#include "cluster_tree/cluster_tree.h"

using namespace HierBEM;
using namespace dealii;

int
main()
{
  /**
   * Initialize deal.ii log stream.
   */
  deallog.pop();
  deallog.depth_console(2);

  /**
   * Generate the grid for a 3D sphere.
   */
  const unsigned int      dim = 3;
  Triangulation<dim, dim> triangulation;
  /**
   * N.B. Use type cast for triangulation to suppress Eclipse editor error
   * prompt.
   */
  GridGenerator::hyper_ball((Triangulation<dim> &)triangulation,
                            Point<3>(0., 0., 0.),
                            2.0,
                            true);
  triangulation.refine_global(1);

  /**
   * Save the mesh to a file for visualization.
   */
  GridOut       grid_out;
  std::ofstream mesh_file("ball.msh");
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Create a Lagrangian finite element.
   */
  const unsigned int fe_order = 1;
  FE_Q<dim, dim>     fe(fe_order);

  /**
   * Create a DoFHandler, which is associated with the triangulation and
   * distributed with the finite element.
   */
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /**
   * Create a mapping object, which is required in generating the map from
   * DoF indices to support points.
   */
  const MappingQ<dim, dim> mapping(fe_order);

  /**
   * Generate a list of all DoF indices.
   */
  std::vector<types::global_dof_index> dof_indices(dof_handler.n_dofs());
  types::global_dof_index              counter = 0;
  for (auto &dof_index : dof_indices)
    {
      dof_index = counter;
      counter++;
    }

  /**
   * Get the spatial coordinates of the support points associated with DoF
   * indices.
   */
  std::vector<Point<dim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);

  /**
   * Initialize a cluster tree for all the DoF indices.
   */
  const unsigned int n_min = 4;
  ClusterTree<dim>   cluster_tree(dof_indices, all_support_points, n_min);

  /**
   * Partition the cluster tree.
   */
  cluster_tree.partition(all_support_points);

  /**
   * Print the coordinates of all support points.
   */
  deallog << "=== Support point coordinates ===\n";
  for (auto &point : all_support_points)
    {
      deallog << point << "\n";
    }

  /**
   * Print the whole cluster tree.
   */
  deallog << "=== Cluster tree ===\n";
  deallog << cluster_tree << std::endl;

  /**
   * Compute the memory consumption.
   */
  deallog << "Memory consumption of all clusters: "
          << cluster_tree.memory_consumption_of_all_clusters() << "\n";
  deallog << "Memory consumption: " << cluster_tree.memory_consumption()
          << std::endl;

  /**
   * Export the cluster tree as a directional graph.
   */
  std::ofstream graph("cluster-tree.puml");
  cluster_tree.print_tree_info_as_dot(graph);
  graph.close();
}
