/**
 * \file hmatrix-transpose-forward-substitution-matrix-valued.cc
 * \brief Verify matrix-valued forward substitution of the transpose of a upper
 * triangular \hmatrix.
 *
 * \details The problem to be solved is \f$XU=Z\f$, where both \f$X\f$ and
 * \f$Z\f$ are \hmatrices, which have a same \bct structure. However, this
 * structure can be different from that of \f$U\f$. The \bct partition structure
 * is fine non-tensor product.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-10-29
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
run_hmatrix_transpose_forward_substitution_matrix_valued();

extern void
run_hmatrix_transpose_forward_substitution_matrix_valued_in_situ();

static constexpr int FUZZING_TIMES = 5;
TEST_CASE(
  "Solve transposed upper triangular H-matrix using matrix-valued forward substitution",
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
      [&]() { inst.source_file(SOURCE_DIR "/gen_matrices.m"); }());

    run_hmatrix_transpose_forward_substitution_matrix_valued();

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
    out = inst.eval_string("U_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("Z_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("X_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE(
  "Solve transposed upper triangular H-matrix using matrix-valued forward substitution in situ",
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
      [&]() { inst.source_file(SOURCE_DIR "/gen_matrices.m"); }());

    run_hmatrix_transpose_forward_substitution_matrix_valued_in_situ();

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
    out = inst.eval_string("U_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("Z_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("X_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}
