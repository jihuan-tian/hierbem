// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file assign-entity-tag-to-cell.cc
 * @brief Verify @p read_mesh in "grid/grid_in_ext.h", which assigns entity tag
 * as material id to each cell. By disabling orientation checking, it can read a
 * skeleton surface mesh, which is not homeomorphic to a sphere.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-07-28
 */

#include <deal.II/grid/tria.h>

#include <iostream>

#include "grid/grid_in_ext.h"
#include "grid/grid_out_ext.h"
#include "hbem_test_config.h"

using namespace dealii;
using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  Triangulation<2, 3> tria;
  // Read the skeleton mesh.
  read_msh(HBEM_TEST_MODEL_DIR "sphere-immersed-in-two-boxes.msh", tria, false);
  // Write the skeleton mesh as an MSH file, where there are three tags in an
  // element data record: material id, elementary tag and subdomain id. The
  // elementary tag is the same as the material id, so that we can visualize
  // material id directly in Gmsh.
  write_msh_correct(tria, std::cout);

  return 0;
}
