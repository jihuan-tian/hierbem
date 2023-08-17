/**
 * \file get_vertices_in_unit_cell.cc
 * \brief Verify the extraction of vertices in a unit cell of a finite element.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-06-14
 */

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_q.h>

#include <array>
#include <iostream>

#include "bem_tools.h"
#include "debug_tools.h"

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
