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
