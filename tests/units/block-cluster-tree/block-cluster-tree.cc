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
 * \file block-cluster-tree.cc
 * \brief This files verifies the admissible block cluster partition for a
 * mesh.
 *
 * \ingroup test_cases hierarchical_matrices
 * \date 2021-04-28
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

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "cluster_tree/block_cluster_tree.h"
#include "cluster_tree/cluster_tree.h"
#include "cluster_tree/simple_bounding_box.h"
#include "dofs/dof_tools_ext.h"
#include "utilities/debug_tools.h"

using namespace HierBEM;

int
main(int argc, char *argv[])
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

  if (argc > 1)
    {
      /**
       * Read the mesh from a file.
       */
      GridIn<dim, spacedim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::fstream mesh_file(argv[1]);
      grid_in.read_msh(mesh_file);
    }
  else
    {
      std::vector<unsigned int> n_subdivisions{10, 10};
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
    }


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

  /**
   * Get the spatial coordinates of the support points associated with DoF
   * indices.
   */
  std::vector<Point<spacedim>> all_support_points(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping,
                                       dof_handler,
                                       all_support_points);


  /**
   * Calculate the average mesh cell size at each support point.
   */
  std::vector<double> dof_average_cell_size(dof_handler.n_dofs(), 0);
  DoFToolsExt::map_dofs_to_average_cell_size(dof_handler,
                                             dof_average_cell_size);

  /**
   * Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
   * indices.
   */
  const unsigned int    n_min = 4;
  ClusterTree<spacedim> TI(dof_indices,
                           all_support_points,
                           dof_average_cell_size,
                           n_min);

  /**
   * Partition the cluster tree.
   */
  TI.partition(all_support_points, dof_average_cell_size);

  unsigned int dofs_in_leaf_set = 0;
  for (auto node : TI.get_leaf_set())
    {
      dofs_in_leaf_set += node->get_data_reference().get_cardinality();
    }

  deallog << "Total number of DoFs: " << all_support_points.size() << "\n";
  deallog << "Total number of DoFs in cluster tree leaf set: "
          << dofs_in_leaf_set << "\n";

  /**
   * Print the cluster tree.
   */
  deallog << "=== Cluster tree TI ===\n";
  deallog << TI << std::endl;

  /**
   * Create the block cluster tree.
   */
  const double               eta = 10;
  BlockClusterTree<spacedim> block_cluster_tree(TI, TI, eta);

  /**
   * Perform admissible partition on the block cluster tree.
   */
  block_cluster_tree.partition(TI.get_external_to_internal_dof_numbering(),
                               all_support_points);

  /**
   * Print the block cluster tree, even though there is only a root node.
   */
  deallog << "=== Block cluster tree ===\n";
  deallog << block_cluster_tree << "\n";
  deallog << "Memory consumption of all block clusters: "
          << block_cluster_tree.memory_consumption_of_all_block_clusters()
          << "\n";
  deallog << "Memory consumption: " << block_cluster_tree.memory_consumption()
          << std::endl;

  //  /**
  //   * Create a deal.ii vector storing the block cluster levels for all DoFs.
  //   */
  //  Vector<float>     tau_dof_levels;
  //  Vector<float>     sigma_dof_levels;
  //  Vector<float>     tau_block_cluster_indices;
  //  Vector<float>     sigma_block_cluster_indices;
  //  FullMatrix<float> block_cluster_level_matrix;
  //  FullMatrix<float> block_cluster_index_matrix;
  //
  //  tau_dof_levels.reinit(dof_handler.n_dofs());
  //  sigma_dof_levels.reinit(dof_handler.n_dofs());
  //  tau_block_cluster_indices.reinit(dof_handler.n_dofs());
  //  sigma_block_cluster_indices.reinit(dof_handler.n_dofs());
  //  block_cluster_level_matrix.reinit(dof_handler.n_dofs(),
  //  dof_handler.n_dofs());
  //  block_cluster_index_matrix.reinit(dof_handler.n_dofs(),
  //  dof_handler.n_dofs());
  //
  //  /**
  //   * Iterate through the \f$\tau\f$ component of each block cluster and get
  //   the
  //   * level of each DoF in \f$\tau\f$. Then assign the level value to all
  //   DoFs
  //   * contained in the cluster \f$\tau\f$.
  //   */
  //  long unsigned int block_cluster_index = 0;
  //  for (auto block_cluster_node : block_cluster_tree.get_leaf_set())
  //    {
  //      const unsigned int tau_level =
  //        block_cluster_node->get_data_pointer()->get_tau_node()->get_level();
  //      for (types::global_dof_index dof_index :
  //           block_cluster_node->get_data_pointer()
  //             ->get_tau_node()
  //             ->get_data_pointer()
  //             ->get_index_set())
  //        {
  //          tau_dof_levels(dof_index)            = tau_level;
  //          tau_block_cluster_indices(dof_index) = block_cluster_index;
  //        }
  //
  //      const unsigned int sigma_level =
  //        block_cluster_node->get_data_pointer()->get_sigma_node()->get_level();
  //      for (types::global_dof_index dof_index :
  //           block_cluster_node->get_data_pointer()
  //             ->get_sigma_node()
  //             ->get_data_pointer()
  //             ->get_index_set())
  //        {
  //          sigma_dof_levels(dof_index)            = sigma_level;
  //          sigma_block_cluster_indices(dof_index) = block_cluster_index;
  //        }
  //
  //      for (types::global_dof_index tau_dof_index :
  //           block_cluster_node->get_data_pointer()
  //             ->get_tau_node()
  //             ->get_data_pointer()
  //             ->get_index_set())
  //        {
  //          for (types::global_dof_index sigma_dof_index :
  //               block_cluster_node->get_data_pointer()
  //                 ->get_sigma_node()
  //                 ->get_data_pointer()
  //                 ->get_index_set())
  //            {
  //              block_cluster_level_matrix(tau_dof_index, sigma_dof_index) =
  //                tau_level;
  //              block_cluster_index_matrix(tau_dof_index, sigma_dof_index) =
  //                block_cluster_index;
  //            }
  //        }
  //
  //      block_cluster_index++;
  //    }
  //
  //  /**
  //   * Cluster indices in a cluster tree: on level 2.
  //   */
  //  Vector<float> cluster_indices;
  //  cluster_indices.reinit(dof_handler.n_dofs());
  //
  //  unsigned int cluster_index = 0;
  //  for (const auto index : TI.get_root()
  //                            ->Left()
  //                            ->Left()
  //                            ->Left()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Left()
  //                            ->Left()
  //                            ->Right()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Left()
  //                            ->Right()
  //                            ->Left()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Left()
  //                            ->Right()
  //                            ->Right()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Right()
  //                            ->Left()
  //                            ->Left()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Right()
  //                            ->Left()
  //                            ->Right()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Right()
  //                            ->Right()
  //                            ->Left()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  cluster_index++;
  //  for (const auto index : TI.get_root()
  //                            ->Right()
  //                            ->Right()
  //                            ->Right()
  //                            ->get_data_pointer()
  //                            ->get_index_set())
  //    {
  //      cluster_indices(index) = cluster_index;
  //    }
  //
  //  DataOut<dim, DoFHandler<dim, spacedim>> data_out;
  //  data_out.attach_dof_handler(dof_handler);
  //  std::ofstream output("cluster-level.vtk");
  //
  //  data_out.add_data_vector(tau_dof_levels, "tau_cluster_level");
  //  data_out.add_data_vector(sigma_dof_levels, "sigma_cluster_level");
  //  data_out.add_data_vector(tau_block_cluster_indices,
  //                           "tau_block_cluster_index");
  //  data_out.add_data_vector(sigma_block_cluster_indices,
  //                           "sigma_block_cluster_index");
  //  data_out.add_data_vector(cluster_indices, "cluster_index");
  //
  //  data_out.build_patches();
  //  data_out.write_vtk(output);
  //
  //  /**
  //   * Save the block cluster data matrices.
  //   */
  //  std::ofstream level_matrix_file("block-cluster-level.dat");
  //  std::ofstream index_matrix_file("block-cluster-index.dat");
  //  block_cluster_level_matrix.print(level_matrix_file);
  //  block_cluster_index_matrix.print(index_matrix_file, 10);
  //  level_matrix_file.close();
  //  index_matrix_file.close();
}
