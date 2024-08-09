/**
 * \file verify-vertex-order-in-cell-and-fe-support-points.cc
 * \brief Verify the consistency between the order of vertices in a cell and the
 * support points transformed from unit support points via a mapping object.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-07-12
 */

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <cmath>
#include <iostream>

#include "debug_tools.h"
#include "mapping/mapping_q_generic_ext.h"

int
main()
{
  const unsigned int dim           = 2;
  const unsigned int spacedim      = 3;
  const unsigned int fe_order      = 3;
  const unsigned int mapping_order = 1;

  /**
   * Create a single cell mesh.
   */
  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_rectangle(tria,
                                 Point<dim>(1.1, 10.8),
                                 Point<dim>(3.6, 15.2));
  GridTools::rotate(30. / 180. * M_PI, 0, tria);
  GridTools::rotate(70. / 180. * M_PI, 1, tria);
  GridTools::rotate(10. / 180. * M_PI, 2, tria);
  std::ofstream out("one-cell.msh");
  GridOut().write_msh(tria, out);

  /**
   * Output the vertices of the single cell in the default order.
   */
  std::cout << "=== Vertices obtained via CellAccessor ===" << std::endl;
  const auto cell = tria.begin_active();
  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++)
    {
      std::cout << cell->vertex(i) << std::endl;
    }

  /**
   * Define the finite element FE_Q and transform its unit support points to the
   * real cell.
   */
  std::cout
    << "=== Support points obtained via FE_Q in the hierarchic order ==="
    << std::endl;
  FE_Q<dim, spacedim>               fe_q(fe_order);
  MappingQGenericExt<dim, spacedim> mapping(mapping_order);
  const std::vector<Point<dim>>    &fe_q_unit_support_points =
    fe_q.get_unit_support_points();
  for (const auto p : fe_q_unit_support_points)
    {
      std::cout << mapping.transform_unit_to_real_cell(cell, p) << std::endl;
    }

  /**
   * Define the finite element FE_DGQ and transform its unit support points to
   * the real cell.
   */
  std::cout
    << "=== Support points obtained via FE_DGQ in the lexicographic order ==="
    << std::endl;
  FE_DGQ<dim, spacedim>          fe_dgq(fe_order);
  const std::vector<Point<dim>> &fe_dgq_unit_support_points =
    fe_dgq.get_unit_support_points();
  for (const auto p : fe_dgq_unit_support_points)
    {
      std::cout << mapping.transform_unit_to_real_cell(cell, p) << std::endl;
    }

  return 0;
}
