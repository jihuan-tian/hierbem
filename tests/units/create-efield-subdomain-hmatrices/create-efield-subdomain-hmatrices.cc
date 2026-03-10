// Copyright (C) 2024-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file create-efield-subdomain-hmatrices.cc
 * @brief
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2024-08-09
 */
#include <catch2/catch_all.hpp>

#include <fstream>

#include "config_file/cu_related.h"
#include "electric_field/ddm_efield.h"
#include "grid/grid_in_ext.h"
#include "grid/grid_out_ext.h"
#include "hbem_test_config.h"

using namespace dealii;
using namespace HierBEM;
using namespace Catch::Matchers;

TEST_CASE("Create subdomain H-hmatrices", "[ddm_efield]")
{
  deallog.pop();
  deallog.depth_console(0);
  deallog.depth_file(5);
  deallog.attach(std::cout);

  ConfParallelization parallel_params;
  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

  DDMEfield<2, 3, double, double> efield(1,    // fe order for dirichlet space
                                         0,    // fe order for neumann space
                                         32,   // n_min for cluster tree
                                         32,   // n_min for block cluster tree
                                         0.8,  // eta for H-matrix
                                         5,    // max rank for H-matrix
                                         0.01, // aca epsilon for H-matrix
                                         1.0,  // eta for preconditioner
                                         2,    // max rank for preconditioner
                                         0.1   // aca epsilon for preconditioner
  );
  efield.read_subdomain_topology(HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.brep",
                                 HBEM_TEST_MODEL_DIR
                                 "sphere-immersed-in-two-boxes.msh");
  read_msh(HBEM_TEST_MODEL_DIR "sphere-immersed-in-two-boxes.msh",
           efield.get_triangulation(),
           false);
  // At the moment, we manually assign problem parameters.
  efield.initialize_parameters();
  efield.setup_system();
  efield.assemble_system();
  efield.output_results();
}
