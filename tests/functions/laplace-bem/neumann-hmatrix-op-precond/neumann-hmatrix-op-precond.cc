/**
 * \file neumann-hmatrix-op-precond.cc
 * \brief Verify solving the Laplace problem with Neumann boundary condition
 * using H-matrix based BEM. Operator preconditioning is used.
 *
 * \ingroup preconditioner
 * \author Jihuan Tian
 * \date 2025-01-07
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_neumann_hmatrix_op_precond(const unsigned int refinement);

TEST_CASE(
  "Solve Laplace problem with Neumann boundary condition using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_neumann_hmatrix_op_precond(1);

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
