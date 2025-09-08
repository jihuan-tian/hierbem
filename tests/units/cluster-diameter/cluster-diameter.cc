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
 * \file cluster-diameter.cc
 * \brief This files verifies the cluster diameter and pair-wise distance
 * calculation using a 3x3 grid in a square.
 * \ingroup hierarchical_matrices
 * \date 2021-04-25
 * \author Jihuan Tian
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
#include "utilities/debug_tools.h"

int
main()
{
  /**
   * Initialize deal.ii log stream.
   */
  deallog.pop();
  deallog.depth_console(2);

  /**
   * Generate the 3x3 grid in a 2D square.
   */
  const unsigned int           spacedim = 3;
  const unsigned int           dim      = 2;
  Triangulation<dim, spacedim> triangulation;
  std::vector<unsigned int>    n_subdivisions{{3, 3}};
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            n_subdivisions,
                                            Point<dim>(0, 0),
                                            Point<dim>(1, 1));

  /**
   * Save the mesh to a file for visualization.
   */
  GridOut       grid_out;
  std::ofstream mesh_file("square.msh");
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Create the Lagrangian finite element.
   */
  const unsigned int  fe_order = 1;
  FE_Q<dim, spacedim> fe(fe_order);

  /**
   * Create a DoFHandler, which is associated with the triangulation and
   * distributed with the finite element.
   */
  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /**
   * Create the mapping object, which is required in generating the map from
   * DoF indices to support points.
   */
  const MappingQ<dim, spacedim> mapping(fe_order);

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

  std::vector<Point<spacedim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);

  /**
   * Print the coordinates of all support points.
   */
  deallog << "=== Support point coordinates ===\n";
  for (auto &point : all_support_points)
    {
      deallog << point << "\n";
    }

  /**
   * Write DoF indices at each support point.
   */
  print_support_point_info(fe, mapping, dof_handler, "dof_numbering");

  /**
   * Calculate the average mesh cell size at each support point.
   */
  std::vector<double> dof_average_cell_size(dof_handler.n_dofs(), 0);
  std::vector<double> dof_max_cell_size(dof_handler.n_dofs(), 0);
  std::vector<double> dof_min_cell_size(dof_handler.n_dofs(), 0);
  map_dofs_to_average_cell_size(dof_handler, dof_average_cell_size);
  map_dofs_to_max_cell_size(dof_handler, dof_max_cell_size);
  map_dofs_to_min_cell_size(dof_handler, dof_min_cell_size);

  deallog << "=== Average cell size at each support point ===" << std::endl;
  print_vector_values(deallog.get_console(), dof_average_cell_size);
  deallog << "=== Max cell size at each support point ===" << std::endl;
  print_vector_values(deallog.get_console(), dof_max_cell_size);
  deallog << "=== Min cell size at each support point ===" << std::endl;
  print_vector_values(deallog.get_console(), dof_min_cell_size);

  /**
   * Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
   * indices.
   */
  const unsigned int    n_min = 4;
  ClusterTree<spacedim> TI(dof_indices,
                           all_support_points,
                           dof_average_cell_size,
                           n_min);
  ClusterTree<spacedim> TJ(dof_indices,
                           all_support_points,
                           dof_average_cell_size,
                           n_min);

  /**
   * Partition the cluster tree.
   */
  TI.partition(all_support_points, dof_average_cell_size);
  TJ.partition(all_support_points, dof_average_cell_size);

  /**
   * Print the whole cluster trees.
   */
  deallog << "=== Cluster tree T(I) ===\n";
  deallog << TI << "\n";

  deallog << "=== Cluster tree T(J) ===\n";
  deallog << TJ << std::endl;

  /**
   * Get the cluster containing DoF indices [0 1 2 3].
   */
  typename ClusterTree<spacedim>::data_pointer_type cluster1;
  cluster1 = TI.get_root()->Left()->Left()->get_data_pointer();
  deallog << "Cluster: [0 1 2 3]\n";
  deallog << (*cluster1) << std::endl;
  ;

  /**
   * Get the cluster containing DoF indices [10, 11, 14, 15].
   */
  typename ClusterTree<spacedim>::data_pointer_type cluster2;
  cluster2 = TI.get_root()->Right()->Right()->get_data_pointer();
  deallog << "Cluster: [10 11 14 15]\n";
  deallog << (*cluster2) << std::endl;

  deallog << "Uncorrected distance between cluster1 and cluster2: "
          << cluster1->distance_to_cluster((*cluster2), all_support_points)
          << std::endl;
  deallog << "Corrected distance between cluster1 and cluster2: "
          << cluster1->distance_to_cluster((*cluster2),
                                           all_support_points,
                                           dof_average_cell_size)
          << std::endl;
}
