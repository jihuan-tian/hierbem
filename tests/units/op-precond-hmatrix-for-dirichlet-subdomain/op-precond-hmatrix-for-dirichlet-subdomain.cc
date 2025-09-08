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
 * @file op-precond-hmatrix-for-dirichlet-subdomain-.cu
 * @brief Verify building the preconditioner matrix on refined mesh for
 * operator preconditioning on a subdomain used in Laplace Dirichlet problem.
 *
 * @ingroup preconditioners
 * @author Jihuan Tian
 * @date 2025-01-27
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_op_precond_hmatrix_for_dirichlet();

TEST_CASE("Build preconditioner matrix for Laplace Dirichlet on subdomain",
          "[preconditioner]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_op_precond_hmatrix_for_dirichlet();

  try
    {
      inst.source_file(SOURCE_DIR "/process.m");
    }
  catch (...)
    {
      // Ignore errors
    }

  HBEMOctaveValue out;
  out = inst.eval_string("rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-12));
}
