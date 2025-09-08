// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

// File: surface-jacobian-det.cc
// Description: This test calculates the surface Jacobian determinant for
// quadrangular cell. Author: Jihuan Tian Date: 2020-11-04 Copyright (C) 2020
// Jihuan Tian <jihuan_tian@hotmail.com>

#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

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

  // Generate a sphere grid.
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_sphere(triangulation);
  triangulation.refine_global(3);
  GridOut       gridout;
  std::ofstream mesh_file("./surface-jacobian-det-sphere.msh");
  gridout.write_msh(triangulation, mesh_file);

  //  // Generate a square grid.
  //  Triangulation<dim, spacedim> triangulation;
  //  GridGenerator::hyper_cube(triangulation);
  //  triangulation.refine_global(3);
  //  GridOut gridout;
  //  std::ofstream mesh_file("./surface-jacobian-det-square.msh");
  //  gridout.write_msh(triangulation, mesh_file);

  const unsigned int      n_active_cells = triangulation.n_active_cells();
  Vector<double>          cell_surface_jacobian_det(n_active_cells);
  const unsigned int      fe_degree = 2;
  FE_Q<dim, spacedim>     fe(fe_degree);
  MappingQ<dim, spacedim> mapping(fe_degree);

  // Iterative over each cell and evaluate the surface Jacobian determinant at a
  // specified unit cell point.
  Point<2> unit_cell_center(0.5, 0.5);

  unsigned int i = 0;
  for (const auto cell : triangulation.active_cell_iterators())
    {
      std::vector<Point<spacedim>> real_support_points(
        HierBEM::support_points_in_real_cell(
          cell, fe, mapping, fe.get_poly_space_numbering_inverse()));

      cell_surface_jacobian_det(i) =
        HierBEM::surface_jacobian_det(fe,
                                      real_support_points,
                                      unit_cell_center);

      i++;
    }

  cell_surface_jacobian_det.print(deallog.get_console(), 5, true, false);

  return 0;
}
