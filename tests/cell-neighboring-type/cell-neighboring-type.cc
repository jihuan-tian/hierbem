/**
 * \file cell-neighboring-type.cc
 * \brief Verify the detection of cell neighboring type.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2020-11-10
 */

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/full_matrix.templates.h>

#include <laplace_bem.h>

#include <fstream>
#include <iostream>

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Generate a grid.
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  //  GridOut gridout;
  //  std::ofstream mesh_file("./cell-neighboring-type.msh");
  //  gridout.write_msh(triangulation, mesh_file);

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
          IdeoBEM::get_vertex_indices<dim, spacedim>(first_cell));

      types::global_vertex_index j = 0;
      for (const auto second_cell : triangulation.active_cell_iterators())
        {
          std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
            second_cell_vertex_indices(
              IdeoBEM::get_vertex_indices<dim, spacedim>(second_cell));

          std::vector<types::global_vertex_index> vertex_index_intersection;
          vertex_index_intersection.reserve(
            GeometryInfo<dim>::vertices_per_cell);
          cell_neighboring_type_matrix(i, j) =
            IdeoBEM::detect_cell_neighboring_type<dim>(
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
          IdeoBEM::get_vertex_dof_indices<dim, spacedim>(first_cell));

      types::global_vertex_index j = 0;
      for (const auto second_cell : dof_handler.active_cell_iterators())
        {
          std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
            second_cell_vertex_dof_indices(
              IdeoBEM::get_vertex_dof_indices<dim, spacedim>(second_cell));

          std::vector<types::global_dof_index> vertex_dof_index_intersection;
          vertex_dof_index_intersection.reserve(
            GeometryInfo<dim>::vertices_per_cell);
          cell_neighboring_type_matrix(i, j) =
            IdeoBEM::detect_cell_neighboring_type<dim>(
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
