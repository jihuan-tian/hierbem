// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/full_matrix.templates.h>

#include <fstream>
#include <iostream>

#include "laplace/laplace_bem.h"

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
  GridOut       gridout;
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
            HierBEM::cell_distance<dim, spacedim, double>(first_cell,
                                                          second_cell);

          j++;
        }

      i++;
    }

  cell_distance_matrix.print(deallog);

  return 0;
}
