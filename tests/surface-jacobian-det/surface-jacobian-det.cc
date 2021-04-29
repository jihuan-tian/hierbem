// File: surface-jacobian-det.cc
// Description: This test calculates the surface Jacobian determinant for quadrangular cell.
// Author: Jihuan Tian
// Date: 2020-11-04
// Copyright (C) 2020 Jihuan Tian <jihuan_tian@hotmail.com>

#include <deal.II/base/logstream.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
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

  // Generate a sphere grid.
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_sphere(triangulation);
  triangulation.refine_global(3);
  GridOut gridout;
  std::ofstream mesh_file("./surface-jacobian-det-sphere.msh");
  gridout.write_msh(triangulation, mesh_file);

//  // Generate a square grid.
//  Triangulation<dim, spacedim> triangulation;
//  GridGenerator::hyper_cube(triangulation);
//  triangulation.refine_global(3);
//  GridOut gridout;
//  std::ofstream mesh_file("./surface-jacobian-det-square.msh");
//  gridout.write_msh(triangulation, mesh_file);

  const unsigned int n_active_cells = triangulation.n_active_cells();
  Vector<double> cell_surface_jacobian_det(n_active_cells);
  const unsigned int fe_degree = 2;
  FE_Q<dim, spacedim> fe(fe_degree);
  MappingQGeneric<dim, spacedim> mapping(fe_degree);

  // Iterative over each cell and evaluate the surface Jacobian determinant at a specified unit cell point.
  Point<2> unit_cell_center(0.5, 0.5);

  unsigned int i = 0;
  for (const auto cell : triangulation.active_cell_iterators())
  {
    std::vector<Point<spacedim> > real_support_points(LaplaceBEM::support_points_in_real_cell(cell, fe, mapping, fe.get_poly_space_numbering_inverse()));

    cell_surface_jacobian_det(i) =
        LaplaceBEM::surface_jacobian_det(fe, real_support_points, unit_cell_center);

    i++;
  }

  cell_surface_jacobian_det.print(deallog.get_console(), 5, true, false);

  return 0;
}
