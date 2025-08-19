/**
 * \file neumann-full-matrix.cc
 * \brief Verify solving the complex valued Laplace problem with Neumann
 * boundary condition using full matrix based BEM.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2025-08-19
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_neumann_full_matrix_complex();

TEST_CASE(
  "Solve complex valued Laplace problem with Neumann boundary condition using full matrix",
  "[laplace]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_neumann_full_matrix_complex();

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
