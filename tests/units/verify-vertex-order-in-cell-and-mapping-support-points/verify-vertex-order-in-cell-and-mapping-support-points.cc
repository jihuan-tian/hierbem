/**
 * \file verify-vertex-order-in-cell-and-mapping-support-points.cc
 * \brief Verify the consistency between the order of vertices in a cell and the
 * support points in both @p MappingQExt calculated by
 * @p MappingQExt::compute_mapping_support_points
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-07-12
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <cmath>
#include <iostream>

#include "mapping/mapping_q_ext.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int dim           = 2;
  const unsigned int spacedim      = 3;
  const unsigned int mapping_order = 3;

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
   * Define a mapping object and output its support points with respect to the
   * cell.
   */
  std::cout
    << "=== Support points obtained via MappingQExt in the hierarchic order ==="
    << std::endl;
  MappingQExt<dim, spacedim> mapping(mapping_order);
  mapping.compute_mapping_support_points(cell);
  print_vector_values(std::cout, mapping.get_support_points(), "\n", true);

  return 0;
}
