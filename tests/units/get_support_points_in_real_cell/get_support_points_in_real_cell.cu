// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file get_support_points_in_real_cell.cc
 * \brief Verify extracting support points in a real cell.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-06-13
 */

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "bem/bem_tools.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  Triangulation<2, 3> tria;
  GridGenerator::hyper_cube(tria, 0, 1, false);
  GridTools::rotate(45.0 * M_PI / 180.0, 0, tria);
  GridTools::rotate(45.0 * M_PI / 180.0, 1, tria);
  GridTools::rotate(30.0 * M_PI / 180.0, 2, tria);
  GridTools::shift(Point<3>(10.0, 8.0, -7.0), tria);

  std::ofstream mesh_file("square.msh");
  GridOut().write_msh(tria, mesh_file);

  MappingQ<2, 3> mapping(1);

  std::cout << "FE_Q" << std::endl;
  {
    FE_Q<2, 3>            fe(2);
    std::vector<Point<3>> support_points(
      HierBEM::BEMTools::get_lexicographic_support_points_in_real_cell(
        tria.begin_active(), fe, mapping));
    print_vector_values(std::cout, support_points, "\n", true);
  }

  std::cout << "FE_DGQ" << std::endl;
  {
    FE_DGQ<2, 3>          fe(2);
    std::vector<Point<3>> support_points(fe.dofs_per_cell);
    HierBEM::BEMTools::get_lexicographic_support_points_in_real_cell(
      tria.begin_active(), fe, mapping, support_points);
    print_vector_values(std::cout, support_points, "\n", true);
  }

  return 0;
}
