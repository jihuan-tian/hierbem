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

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <fstream>

#include "laplace/laplace_bem.h"
#include "linear_algebra/linalg.h"

using namespace dealii;
using namespace LinAlg;

int
main(int argc, char *argv[])
{
  (void)argc;

  deallog.depth_console(2);
  deallog.pop();

  std::string         mesh_file_name(argv[1]);
  std::fstream        mesh_file(mesh_file_name);
  Triangulation<2, 3> triangulation;
  GridIn<2, 3>        grid_in;
  grid_in.attach_triangulation(triangulation);
  grid_in.read_msh(mesh_file);

  // Iterate over each cell in the triangulation.
  Vector<double> cell_planarity_determinants(triangulation.n_active_cells());
  unsigned int   counter = 0;
  std::array<Point<3>, GeometryInfo<2>::vertices_per_cell> cell_vertices;
  FullMatrix<double> cell_vertex_matrix(4, 4);
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; v++)
        {
          cell_vertices[v] = cell->vertex(v);
        }

      for (unsigned int i = 0; i < 4; i++)
        {
          for (unsigned int j = 0; j < 4; j++)
            {
              if (j == 3)
                {
                  cell_vertex_matrix(i, j) = 1.;
                }
              else
                {
                  cell_vertex_matrix(i, j) = cell_vertices[i](j);
                }
            }
        }

      cell_planarity_determinants(counter) = determinant4x4(cell_vertex_matrix);
      counter++;
    }

  cell_planarity_determinants.print(deallog.get_console(), 10, true, false);
}
