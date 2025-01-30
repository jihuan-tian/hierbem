/**
 * @file op-precond-hmatrix-for-neumann-subdomain-.cu
 * @brief Verify building the preconditioner matrix on refined mesh for
 * operator preconditioning on a subdomain used in Laplace Neumann problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2025-01-27
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

extern void
run_op_precond_hmatrix_for_neumann();

TEST_CASE("Build preconditioner matrix for Laplace Neumann on subdomain",
          "[preconditioner]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  run_op_precond_hmatrix_for_neumann();

  try
    {
      inst.source_file(SOURCE_DIR "/process.m");
    }
  catch (...)
    {
      // Ignore errors
    }

  HBEMOctaveValue out;
  out = inst.eval_string("rel_err");
  REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-12));
}
