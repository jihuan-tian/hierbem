/**
 * \file verify-two-dofhandlers-on-a-same-triangulation.cc
 * \brief Verify the ordering of cell vertices when they are returned from two
 * DoFHandlers on a same triangulation.
 *
 * \ingroup test_cases dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-08
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "bem/bem_tools.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  /**
   * Generate the volume mesh.
   */
  Triangulation<spacedim, spacedim> volume_triangulation;
  GridGenerator::hyper_ball(volume_triangulation,
                            Point<spacedim>(0, 0, 0),
                            1.0,
                            true);
  volume_triangulation.refine_global(1);

  /**
   * Extract the boundary mesh.
   */
  Triangulation<dim, spacedim> surface_triangulation;
  GridGenerator::extract_boundary_mesh(volume_triangulation,
                                       surface_triangulation);

  /**
   * Create two finite elements which are to be associated with a same
   * triangulation.
   */
  FE_Q<dim, spacedim>   fe_q(2);
  FE_DGQ<dim, spacedim> fe_dgq(2);

  /**
   * Create and initialize two @p DoFHandlers.
   */
  DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
  dof_handler1.initialize(surface_triangulation, fe_q);
  dof_handler2.initialize(surface_triangulation, fe_dgq);

  /**
   * Iterate over each cell in the first @p DoFHandler and check if
   * 1. the ways for enumeration of cells in the triangulation for the two
   * @p DoFHandlers are the same.
   * 2. the ways for enumeration of vertices in each cell for the two
   * @p DoFHandlers are the same. The
   */
  typename DoFHandler<dim, spacedim>::cell_iterator cell_iter_for_fe_q =
    dof_handler1.begin_active();
  typename DoFHandler<dim, spacedim>::cell_iterator cell_iter_for_fe_dgq =
    dof_handler2.begin_active();

  FullMatrix<int> dof_handler1_vertex_indices(
    dof_handler1.get_triangulation().n_active_cells(),
    GeometryInfo<dim>::vertices_per_cell);
  FullMatrix<int> dof_handler2_vertex_indices(
    dof_handler2.get_triangulation().n_active_cells(),
    GeometryInfo<dim>::vertices_per_cell);

  std::array<types::global_vertex_index, GeometryInfo<dim>::vertices_per_cell>
    vertex_indices_in_cell;

  unsigned int row_counter = 0;
  for (; cell_iter_for_fe_q != dof_handler1.end();
       cell_iter_for_fe_q++, cell_iter_for_fe_dgq++, row_counter++)
    {
      vertex_indices_in_cell =
        HierBEM::BEMTools::get_vertex_indices<dim, spacedim>(
          cell_iter_for_fe_q);

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        {
          dof_handler1_vertex_indices(row_counter, v) =
            vertex_indices_in_cell[v];
        }

      vertex_indices_in_cell =
        HierBEM::BEMTools::get_vertex_indices<dim, spacedim>(
          cell_iter_for_fe_dgq);

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
        {
          dof_handler2_vertex_indices(row_counter, v) =
            vertex_indices_in_cell[v];
        }
    }

  print_matrix_to_mat(std::cout,
                      "dof_handler1_vertex_indices",
                      dof_handler1_vertex_indices,
                      5,
                      false,
                      8,
                      "0");

  print_matrix_to_mat(std::cout,
                      "dof_handler2_vertex_indices",
                      dof_handler2_vertex_indices,
                      5,
                      false,
                      8,
                      "0");

  return 0;
}
