// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file build-cluster-tree.cc
 * @brief Example for building a cluster tree.
 * @ingroup examples
 * @author Jihuan Tian
 * @date 2025-10-02
 */

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iostream>

#include "cluster_tree/cluster_tree.h"

using namespace HierBEM;
using namespace dealii;

int
main()
{
  // Generate a triangulation for a unit sphere.
  const unsigned int    dim      = 2;
  const unsigned int    spacedim = 3;
  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);
  const unsigned int    refinement = 2;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(refinement);

  // Save the mesh to a file for visualization.
  GridOut       grid_out;
  std::ofstream mesh_file("sphere.msh");
  grid_out.write_msh(tria, mesh_file);


  // Create a Lagrangian finite element and a DoF handler.
  const unsigned int        fe_order = 1;
  FE_Q<dim, spacedim>       fe(fe_order);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create a mapping object for transforming unit support points in the unit
  // cell to all real cells.
  const unsigned int            mapping_order = 2;
  const MappingQ<dim, spacedim> mapping(mapping_order);
  // Get the coordinates for all support points.
  std::vector<Point<spacedim>> support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  // Generate a list of DoF indices starting from zero.
  std::vector<types::global_dof_index> dof_indices(dof_handler.n_dofs());
  for (types::global_dof_index d = 0; d < dof_indices.size(); d++)
    dof_indices[d] = d;


  // Create a cluster tree for all the DoF indices.
  const unsigned int    n_min = 4;
  ClusterTree<spacedim> cluster_tree(dof_indices, support_points, n_min);
  // Partition the cluster tree.
  cluster_tree.partition(support_points);

  // Export the cluster tree as a directed graph.
  std::ofstream graph("cluster-tree.puml");
  cluster_tree.print_tree_info_as_dot(graph);
  graph.close();
}
