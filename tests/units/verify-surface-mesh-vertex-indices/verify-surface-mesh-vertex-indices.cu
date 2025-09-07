/**
 * \file verify-surface-mesh-vertex-indices.cc
 * \brief Verify if the vertex indices in a surface mesh that is obtained from
 * @p extract_boundary_mesh are consistent with vertex indices in the original
 * volume mesh.
 *
 * \ingroup test_cases dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-06
 */

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <map>

#include "bem/bem_tools.h"
#include "utilities/debug_tools.h"

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  /**
   * Generate the volume mesh.
   */
  Triangulation<spacedim, spacedim> volume_triangulation;
  GridGenerator::hyper_ball(volume_triangulation,
                            Point<spacedim>(0, 0, 0),
                            1.0,
                            true);
  volume_triangulation.refine_global(2);

  /**
   * Extract surface mesh from the volume mesh.
   */
  Triangulation<dim, spacedim> triangulation;
  auto                         cell_map =
    GridGenerator::extract_boundary_mesh(volume_triangulation, triangulation);

  /**
   * Print vertex indices in each surface cell.
   */
  std::cout << "Vertex indices in surface mesh | Vertex indices in volume mesh"
            << std::endl;
  for (const auto &e : cell_map)
    {
      /**
       * \alert{C++ compiler cannot deduct the correct candidate function
       * @p get_vertex_indices. Therefore, we specify the template arguments
       * explicitly.}
       */
      auto vertex_indices_in_surface_mesh =
        HierBEM::BEMTools::get_vertex_indices<dim, spacedim>(e.first);
      auto vertex_indices_in_face_of_volume_mesh =
        HierBEM::BEMTools::get_face_vertex_indices<spacedim, spacedim>(
          e.second);

      print_vector_values(std::cout,
                          vertex_indices_in_surface_mesh,
                          ",",
                          false);
      std::cout << "|";
      print_vector_values(std::cout,
                          vertex_indices_in_face_of_volume_mesh,
                          ",",
                          false);
      std::cout << std::endl;
    }

  return 0;
}
