/**
 * \file cluster-tree.cc
 * This file verifies the ClusterTree class.
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <simple_bounding_box.h>

#include <fstream>

#include "cluster_tree.h"

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
  GridGenerator::hyper_ball(triangulation, Point<3>(0., 0., 0.), 2.0, true);
  // triangulation.refine_global(2);

  /**
   * Save the mesh to a file for visualization.
   */
  GridOut       grid_out;
  std::ofstream mesh_file("ball.msh");
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Create a high order Lagrangian finite element.
   */
  const unsigned int fe_order = 1;
  FE_Q<dim, dim>     fe(fe_order);

  /**
   * Create a DoFHandler, which is associated with the triangulation and
   * distributed with the finite element.
   */
  DoFHandler<dim, dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /**
   * Create a 2nd order mapping, which is required in generating the map from
   * DoF indices to support points.
   */
  const MappingQGeneric<dim, dim> mapping(fe_order);

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
}
