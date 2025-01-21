/**
 * \file dirichlet-hmatrix-two-spheres-op-precond.cc
 * \brief
 * \ingroup preconditioner
 *
 * \author Jihuan Tian
 * \date 2022-09-23
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_dirichlet_hmatrix_two_spheres();

TEST_CASE(
  "Solve Laplace problem with Dirichlet boundary condition for the two sphere model using operator preconditioning",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_dirichlet_hmatrix_two_spheres();

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
