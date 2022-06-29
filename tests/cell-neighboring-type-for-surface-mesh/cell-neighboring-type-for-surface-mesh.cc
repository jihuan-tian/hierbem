/**
 * \file cell-neighboring-type-for-surface-mesh.cc
 * \brief Verify the detection of cell neighboring type, where the surface
 * triangulation used is extracted from a volume mesh.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-06-06
 */

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/full_matrix.templates.h>

#include <fstream>
#include <iostream>

#include "bem_tools.h"
#include "debug_tools.h"

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

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
  volume_triangulation.refine_global(2);

  /**
   * Save the mesh to file.
   */
  //  std::ofstream mesh_file("ball.msh");
  //  GridOut().write_msh(volume_triangulation, mesh_file);

  std::cout << "=== Volume triangulation info ===" << std::endl;
  print_triangulation_info(std::cout, volume_triangulation);

  /**
   * Extract surface mesh from the volume mesh.
   */
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::extract_boundary_mesh(volume_triangulation, triangulation);

  std::cout << "=== Surface triangulation info ===" << std::endl;
  print_triangulation_info(std::cout, triangulation);

  // Calculate cell neighboring type using vertex indices.
  const unsigned int       n_active_cells = triangulation.n_active_cells();
  FullMatrix<unsigned int> cell_neighboring_type_matrix(n_active_cells,
                                                        n_active_cells);
  // Iterative over each pair of cells.
  types::global_vertex_index i = 0;

  for (const auto first_cell : triangulation.active_cell_iterators())
    {
      std::array<types::global_vertex_index,
                 GeometryInfo<dim>::vertices_per_cell>
        first_cell_vertex_indices(
          IdeoBEM::BEMTools::get_vertex_indices<dim, spacedim>(first_cell));

      types::global_vertex_index j = 0;
      for (const auto second_cell : triangulation.active_cell_iterators())
        {
          std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
            second_cell_vertex_indices(
              IdeoBEM::BEMTools::get_vertex_indices<dim, spacedim>(
                second_cell));

          std::vector<types::global_vertex_index> vertex_index_intersection;
          vertex_index_intersection.reserve(
            GeometryInfo<dim>::vertices_per_cell);
          cell_neighboring_type_matrix(i, j) =
            IdeoBEM::BEMTools::detect_cell_neighboring_type<dim>(
              first_cell_vertex_indices,
              second_cell_vertex_indices,
              vertex_index_intersection);

          j++;
        }

      i++;
    }

  deallog << "Calculate cell neighboring type using vertex indices..."
          << std::endl;
  cell_neighboring_type_matrix.print(deallog);

  // Calculate cell neighboring type using vertex dof indices.
  FE_Q<dim, spacedim>       fe(2);
  DoFHandler<dim, spacedim> dof_handler;
  dof_handler.initialize(triangulation, fe);

  cell_neighboring_type_matrix = 0.;
  i                            = 0;
  for (const auto first_cell : dof_handler.active_cell_iterators())
    {
      std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
        first_cell_vertex_dof_indices(
          IdeoBEM::BEMTools::get_vertex_dof_indices<dim, spacedim>(first_cell));

      types::global_vertex_index j = 0;
      for (const auto second_cell : dof_handler.active_cell_iterators())
        {
          std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
            second_cell_vertex_dof_indices(
              IdeoBEM::BEMTools::get_vertex_dof_indices<dim, spacedim>(
                second_cell));

          std::vector<types::global_dof_index> vertex_dof_index_intersection;
          vertex_dof_index_intersection.reserve(
            GeometryInfo<dim>::vertices_per_cell);
          cell_neighboring_type_matrix(i, j) =
            IdeoBEM::BEMTools::detect_cell_neighboring_type<dim>(
              first_cell_vertex_dof_indices,
              second_cell_vertex_dof_indices,
              vertex_dof_index_intersection);

          j++;
        }

      i++;
    }

  deallog << "Calculate cell neighboring type using vertex dof indices..."
          << std::endl;
  cell_neighboring_type_matrix.print(deallog);

  return 0;
}
