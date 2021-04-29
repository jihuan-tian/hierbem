#include <deal.II/base/logstream.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/full_matrix.templates.h>
#include <laplace_bem.h>

#include <fstream>
#include <iostream>

int main()
{
  deallog.pop ();
  deallog.depth_console(2);

  const unsigned int dim = 2;
  const unsigned int spacedim = 3;

  // Generate a grid.
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  GridOut gridout;
  std::ofstream mesh_file("./cell-distance.msh");
  gridout.write_msh(triangulation, mesh_file);

  const unsigned int n_active_cells = triangulation.n_active_cells();
  FullMatrix<double> cell_distance_matrix(n_active_cells, n_active_cells);
  // Iterative over each pair of cells.
  types::global_vertex_index i = 0;

  for (const auto first_cell : triangulation.active_cell_iterators())
  {
    types::global_vertex_index j = 0;
    for (const auto second_cell : triangulation.active_cell_iterators())
    {
      cell_distance_matrix(i, j) =
          LaplaceBEM::cell_distance<dim, spacedim, double>(first_cell, second_cell);

      j++;
    }

    i++;
  }

  cell_distance_matrix.print(deallog);

  return 0;
}
