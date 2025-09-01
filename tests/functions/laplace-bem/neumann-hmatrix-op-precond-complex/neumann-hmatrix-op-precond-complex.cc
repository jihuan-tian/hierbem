/**
 * \file neumann-hmatrix-op-precond-complex.cc
 * \brief Verify solving the complex valued Laplace problem with Neumann
 * boundary condition using H-matrix based BEM. Operator preconditioning is
 * used.
 *
 * \ingroup preconditioner
 * \author Jihuan Tian
 * \date 2025-08-19
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix_vmult_strategy.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_neumann_hmatrix_op_precond_complex(
  const unsigned int             refinement,
  const IterativeSolverVmultType vmult_type);

TEST_CASE(
  "Solve Laplace problem with complex valued Neumann boundary condition using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  SECTION("serial recursive vmult")
  {
    run_neumann_hmatrix_op_precond_complex(
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

  SECTION("serial iterative vmult")
  {
    run_neumann_hmatrix_op_precond_complex(
      1, IterativeSolverVmultType::SerialIterative);

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
    run_neumann_hmatrix_op_precond_complex(
      1, IterativeSolverVmultType::TaskParallel);

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
