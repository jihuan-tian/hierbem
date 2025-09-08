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
 * @file create-efield-subdomains.cu
 * @brief
 *
 * @ingroup test_cases
 * @author
 * @date 2024-08-06
 */

#include "electric_field/ddm_efield.h"
#include "hbem_test_config.h"

using namespace HierBEM;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  DDMEfield<2, 3> efield;
  efield.read_subdomain_topology(HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.brep",
                                 HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.msh");
  // At the moment, we manually assign problem parameters.
  efield.initialize_parameters();

  efield.create_efield_subdomains_and_surfaces();

  return 0;
}
