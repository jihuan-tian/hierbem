/**
 * \file get_unit_support_points.cc
 * \brief Verify extracting the unit support points in a finite element.
 *
 * \ingroup testers support_points_manip
 * \author Jihuan Tian
 * \date 2022-06-13
 */

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <iostream>
#include <vector>

#include "bem_tools.h"
#include "debug_tools.h"

using namespace dealii;

int
main()
{
  std::cout << "FE_Q"
            << "\n";
  {
    FE_Q<2, 3> fe(2);

    print_vector_values<std::vector<Point<2>>>(
      std::cout,
      IdeoBEM::BEMTools::get_lexicographic_unit_support_points(fe),
      "\n",
      true);

    std::vector<Point<2>> unit_support_points;
    unit_support_points =
      IdeoBEM::BEMTools::get_lexicographic_unit_support_points(fe);
    print_vector_values(std::cout, unit_support_points, "\n", true);
  }


  std::cout << "FE_DGQ"
            << "\n";
  {
    FE_DGQ<2, 3> fe(2);

    print_vector_values<std::vector<Point<2>>>(
      std::cout,
      IdeoBEM::BEMTools::get_lexicographic_unit_support_points(fe),
      "\n",
      true);

    std::vector<Point<2>> unit_support_points(fe.dofs_per_cell);
    IdeoBEM::BEMTools::get_lexicographic_unit_support_points(
      fe, unit_support_points);
    print_vector_values(std::cout, unit_support_points, "\n", true);
  }


  return 0;
}
