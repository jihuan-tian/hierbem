// Copyright (C) 2021-2023 Jihuan Tian <jihuan_tian@hotmail.com>
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

using namespace dealii;

int
main(int argc, char *argv[])
{
  (void)argc;

  deallog.depth_console(2);
  deallog.pop();

  std::string         mesh_file_name(argv[1]);
  Triangulation<2, 3> triangulation;
  GridIn<2, 3>        grid_in;
  grid_in.attach_triangulation(triangulation);
  std::fstream mesh_file(mesh_file_name);
  grid_in.read_msh(mesh_file);

  unsigned int cell_counter = 0;
  deallog << "Cell_index Vertex_index X Y Z" << std::endl;
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; i++)
        {
          deallog << cell_counter << " " << cell->vertex_index(i) << " "
                  << cell->vertex(i) << " " << std::endl;
        }

      cell_counter++;
    }
}
