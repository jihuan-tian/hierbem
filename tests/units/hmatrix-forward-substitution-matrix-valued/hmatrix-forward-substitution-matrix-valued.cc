// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file hmatrix-forward-substitution-matrix-valued.cc
 * \brief Verify matrix-valued forward substitution of a lower triangular
 * \hmatrix.
 *
 * \details The problem to be solved is \f$LX=Z\f$, where both \f$X\f$ and
 * \f$Z\f$ are \hmatrices, which have a same \bct structure. However, this
 * structure can be different from that of \f$L\f$. The \bct partition structure
 * is fine non-tensor product.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-24
 */

#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

#include "hbem_octave_wrapper.h"
#include "hbem_test_config.h"
#include "hbem_test_utils.h"

using namespace Catch::Matchers;
using namespace HierBEM;

// XXX Extracted all HierBEM logic into a standalone source to prevent
// Matrix/SparseMatrix data type conflicts
extern void
run_hmatrix_forward_substitution_matrix_valued();

extern void
run_hmatrix_forward_substitution_matrix_valued_in_situ();

static constexpr int FUZZING_TIMES = 5;
TEST_CASE(
  "Solve lower triangular H-matrix using matrix-valued forward substitution",
  "[hmatrix]")
{
  // Create a unique working directory for each test case
  volatile HBEMTestScopedDirectory scoped_dir;

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

    REQUIRE_NOTHROW([&]() { inst.source_file(SOURCE_DIR "/gen_matrix.m"); }());

    run_hmatrix_forward_substitution_matrix_valued();

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
    out = inst.eval_string("L_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("Z_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("X_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE(
  "Solve lower triangular H-matrix using matrix-valued forward substitution in situ",
  "[hmatrix]")
{
  // Create a unique working directory for each test case
  volatile HBEMTestScopedDirectory scoped_dir;

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

    REQUIRE_NOTHROW([&]() { inst.source_file(SOURCE_DIR "/gen_matrix.m"); }());

    run_hmatrix_forward_substitution_matrix_valued_in_situ();

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
    out = inst.eval_string("L_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("Z_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("X_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}
