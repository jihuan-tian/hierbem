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
 * @file read-1d-skeleton-mesh.cc
 * @brief Verify reading 1D skeleton mesh, which is used in 2D BEM.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-08-05
 */

#include "electric_field/ddm_efield.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"

using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  DDMEfield<1, 3> efield;
  read_msh(HBEM_TEST_MODEL_DIR
           "circle-immersed-in-two-squares-different-surface-orientations.msh",
           efield.get_triangulation(),
           false);

  return 0;
}
