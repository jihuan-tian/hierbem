/**
 * \file verify-surface-mesh-refined-with-volume-mesh.cc
 * \brief Verify if the extracted surface mesh by @p extract_boundary_mesh can
 * refine automatically with the volume mesh. The conclusion is the surface mesh
 * will not be refined auotmatically with the volume mesh.
 *
 * \ingroup test_cases dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-06
 */

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>

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

  std::cout << "=== Volume triangulation info ===" << std::endl;
  print_triangulation_info(std::cout, volume_triangulation);

  /**
   * Extract surface mesh from the volume mesh.
   */
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::extract_boundary_mesh(volume_triangulation, triangulation);

  std::cout << "=== Surface triangulation info ===" << std::endl;
  print_triangulation_info(std::cout, triangulation);

  /**
   * Refine the volume mesh and check if the surface mesh is refined along with
   * its refinement.
   */
  for (unsigned int i = 0; i < 4; i++)
    {
      volume_triangulation.refine_global(1);

      std::cout << "=== Volume triangulation info ===" << std::endl;
      print_triangulation_info(std::cout, volume_triangulation);

      std::cout << "=== Surface triangulation info ===" << std::endl;
      print_triangulation_info(std::cout, triangulation);
    }
}
