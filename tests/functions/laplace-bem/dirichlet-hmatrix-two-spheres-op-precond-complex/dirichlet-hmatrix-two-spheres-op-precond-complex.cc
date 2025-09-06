/**
 * \file dirichlet-hmatrix-two-spheres-op-precond-complex.cc
 * \brief
 * \ingroup preconditioner
 *
 * \author Jihuan Tian
 * \date 2025-09-06
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"
#include "hmatrix/hmatrix_vmult_strategy.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_dirichlet_hmatrix_two_spheres_op_precond_complex(
  const IterativeSolverVmultType vmult_type);

TEST_CASE(
  "Solve Laplace problem with complex valued Dirichlet boundary condition for the two sphere model using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  SECTION("serial recursive vmult")
  {
    run_dirichlet_hmatrix_two_spheres_op_precond_complex(
      IterativeSolverVmultType::SerialRecursive);

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

  SECTION("serial iterative vmult")
  {
    run_dirichlet_hmatrix_two_spheres_op_precond_complex(
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
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-3));

    out = inst.eval_string("solution_inf_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-3));
  }

  SECTION("task parallel vmult")
  {
    run_dirichlet_hmatrix_two_spheres_op_precond_complex(
      IterativeSolverVmultType::TaskParallel);

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
}
