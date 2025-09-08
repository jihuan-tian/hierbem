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
 * \file dirichlet-full-matrix-complex.cc
 * \brief Verify solving the complex valued Laplace problem with Dirichlet
 * boundary condition using full matrix based BEM.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2025-05-12
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_dirichlet_full_matrix_complex();

TEST_CASE(
  "Solve Laplace problem with complex valued Dirichlet boundary condition using full matrix",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_dirichlet_full_matrix_complex();

  try
    {
      inst.source_file(SOURCE_DIR "/process.m");
    }
  catch (...)
    {
      // Ignore errors
    }

  // Check relative error
  HBEMOctaveValue out;
  out = inst.eval_string("solution_l2_rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

  out = inst.eval_string("solution_inf_rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
}
