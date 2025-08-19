/**
 * \file laplace-bem-mixed-hmatrix-op-precond-complex.cc
 * \brief Verify solve complex valued Laplace mixed boundary value problem using
 * \hmat based BEM. Operator preconditioning is used.
 *
 * \ingroup preconditioner
 * \author Jihuan Tian
 * \date 2025-08-19
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_mixed_hmatrix_op_precond_complex();

TEST_CASE(
  "Solve complex valued Laplace problem with mixed boundary condition using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_mixed_hmatrix_op_precond_complex();

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
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-3));

  out = inst.eval_string("solution_inf_rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-3));
}
