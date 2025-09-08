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
 * \file get_vertices_in_unit_cell.cc
 * \brief Verify the extraction of vertices in a unit cell of a finite element.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-06-14
 */

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_q.h>

#include <array>
#include <iostream>

#include "bem/bem_tools.h"
#include "utilities/debug_tools.h"

int
main()
{
  FE_Q<2, 3> fe(2);

  print_vector_values(
    std::cout,
    HierBEM::BEMTools::get_vertices_from_lexicographic_unit_support_points(
      fe, false),
    "\n",
    true);
  print_vector_values(
    std::cout,
    HierBEM::BEMTools::get_vertices_from_lexicographic_unit_support_points(
      fe, true),
    "\n",
    true);

  std::array<Point<2>, GeometryInfo<2>::vertices_per_cell> vertices;
  HierBEM::BEMTools::get_vertices_from_lexicographic_unit_support_points(
    fe, vertices, false);
  print_vector_values(std::cout, vertices, "\n", true);
  HierBEM::BEMTools::get_vertices_from_lexicographic_unit_support_points(
    fe, vertices, true);
  print_vector_values(std::cout, vertices, "\n", true);

  return 0;
}
