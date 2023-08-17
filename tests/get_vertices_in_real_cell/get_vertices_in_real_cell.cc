/**
 * \file get_vertices_in_real_cell.cc
 * \brief Verify the extraction of vertices in the real cell.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-06-14
 */

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <array>
#include <iostream>

#include "bem_tools.h"
#include "debug_tools.h"

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

  MappingQGeneric<2, 3> mapping(1);

  {
    FE_DGQ<2, 3> fe(2);
    std::cout << fe.get_name() << "\n";
    print_vector_values(
      std::cout,
      HierBEM::BEMTools::
        get_vertices_from_lexicographic_support_points_in_real_cell(
          tria.begin_active(), fe, mapping, false),
      "\n",
      true);
  }

  {
    FE_DGQ<2, 3> fe(2);
    std::cout << fe.get_name() << "\n";
    std::array<Point<3>, GeometryInfo<2>::vertices_per_cell> vertices;
    HierBEM::BEMTools::
      get_vertices_from_lexicographic_support_points_in_real_cell(
        tria.begin_active(), fe, mapping, vertices, true);
    print_vector_values(std::cout, vertices, "\n", true);
  }
}
