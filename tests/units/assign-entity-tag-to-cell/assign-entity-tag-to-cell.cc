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
 * @brief
 *
 * @ingroup test_cases
 * @author
 * @date 2024-07-28
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

  DDMEfield<2, 3> efield;
  read_msh(HBEM_TEST_MODEL_DIR "sphere-immersed-in-two-boxes.msh",
           efield.get_triangulation(),
           false);

  return 0;
}
