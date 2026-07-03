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
 * \file dirichlet-hmatrix-op-precond.cc
 * \brief Verify solving the Laplace problem with Dirichlet boundary condition
 * using H-matrix based BEM. Operator preconditioning is used.
 *
 * \ingroup preconditioners
 * \author Jihuan Tian
 * \date 2024-12-17
 */

#include <catch2/catch_all.hpp>
#include <rfl.hpp>
#include <rfl/json.hpp>
#include <rfl/toml.hpp>

#include <string>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix_vmult_strategy.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_dirichlet_hmatrix_op_precond(
  const unsigned int             refinement,
  const IterativeSolverVmultType vmult_type,
  const bool                     cpu_serial_without_producer_consumer = false);

TEST_CASE(
  "Solve Laplace problem with Dirichlet boundary condition using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  SECTION("serial recursive vmult")
  {
    SECTION("cpu serial without producer and consumer")
    {
      run_dirichlet_hmatrix_op_precond(
        1, IterativeSolverVmultType::SerialRecursive, true);

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

    SECTION("with producer and consumer")
    {
      run_dirichlet_hmatrix_op_precond(
        1, IterativeSolverVmultType::SerialRecursive);

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
  }

  SECTION("serial iterative vmult")
  {
    run_dirichlet_hmatrix_op_precond(1,
                                     IterativeSolverVmultType::SerialIterative);

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

  SECTION("task parallel vmult")
  {
    run_dirichlet_hmatrix_op_precond(1, IterativeSolverVmultType::TaskParallel);

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
}
