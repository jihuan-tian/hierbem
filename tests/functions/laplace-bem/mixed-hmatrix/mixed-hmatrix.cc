/**
 * \file laplace-bem-mixed-hmatrix.cc
 * \brief Verify solve Laplace mixed boundary value problem using \hmat.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-11-21
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_mixed_hmatrix();

TEST_CASE("Solve Laplace problem with mixed boundary condition", "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_mixed_hmatrix();

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
