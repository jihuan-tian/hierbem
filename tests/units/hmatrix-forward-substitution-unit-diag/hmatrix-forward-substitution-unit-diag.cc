/**
 * \file hmatrix-forward-substitution-unit-diag.cc
 * \brief Verify forward substitution of a unit lower triangle \hmatrix. The
 * \bct partition structure is fine non-tensor product.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-10-21
 */

#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"

using namespace Catch::Matchers;
using namespace HierBEM;

// XXX Extracted all HierBEM logic into a standalone source to prevent
// Matrix/SparseMatrix data type conflicts
extern void
run_hmatrix_forward_substitution_unit_diag();

static constexpr int FUZZING_TIMES = 5;
TEST_CASE(
  "Solve normalized lower triangular H-matrix using forward substitution",
  "[hmatrix]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");

  auto trial_no = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    // Predefine Octave and C++ random seed to make tests repeatable
    int rng_seed = 1234567 + trial_no * 7;

    // Initialize random seed.
    std::ostringstream oss;
    oss << "rand('seed'," << rng_seed << ");\n";
    oss << "randn('seed'," << rng_seed << ");";
    inst.eval_string(oss.str());

    REQUIRE_NOTHROW(
      [&]() { inst.source_file(SOURCE_DIR "/gen_lower_triangle_mat.m"); }());

    run_hmatrix_forward_substitution_unit_diag();

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
    out = inst.eval_string("hmat_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("x_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}
