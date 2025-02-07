/**
 * \file laplace-bem-mixed-hmatrix-op-precond.cc
 * \brief Verify solve Laplace mixed boundary value problem using \hmat based
 * BEM. Operator preconditioning is used.
 *
 * \ingroup preconditioner
 * \author Jihuan Tian
 * \date 2025-02-04
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_mixed_hmatrix_op_precond();

TEST_CASE(
  "Solve Laplace problem with mixed boundary condition using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_mixed_hmatrix_op_precond();

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
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-2));

  out = inst.eval_string("solution_inf_rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-2));
}
