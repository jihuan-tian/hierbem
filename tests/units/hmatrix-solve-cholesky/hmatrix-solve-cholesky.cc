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
 * \file hmatrix-solve-cholesky.cc
 * \brief Verify Cholesky factorization of a positive definite and symmetric
 * \hmatrix and solve this matrix using forward and backward substitution.
 *
 * \details In this tester, the property of the \hmatrix before factorization is
 * set to @p symmetric and the property of the result \hmatrix is set to
 * @p lower_triangular. \alert{If there is no special treatment as that proposed
 * by Bebendorf, the approximation of the original full matrix using \hmatrix
 * must be good enough so that the positive definiteness of the original matrix
 * is preserved and Cholesky factorization is applicable.}
 *
 * \ingroup test_cases hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-13
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
run_hmatrix_solve_cholesky();

extern void
run_hmatrix_solve_cholesky_in_situ();

static constexpr int FUZZING_TIMES = 5;
TEST_CASE("H-matrix solve equations by Cholesky factorization", "[hmatrix]")
{
  // Create a unique working directory for each test case
  volatile HBEMTestScopedDirectory scoped_dir;

  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  auto trial_no = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    // Predefine Octave and C++ random seed to make tests repeatable
    int rng_seed = 1234567 + trial_no * 7;
    // TODO `src/aca_plus.cu` and `include/aca_plus.hcu` use a
    // std::random_device for hardware seeding, need a mechansim to
    // set the seed.

    std::ostringstream oss;
    oss << "rand('seed'," << rng_seed << ");\n";
    oss << "randn('seed'," << rng_seed << ");";
    inst.eval_string(oss.str());

    // Execute script `gen_matrix.m` to generate M.dat and b.dat
    REQUIRE_NOTHROW([&]() { inst.source_file(SOURCE_DIR "/gen_matrix.m"); }());

    // Run solving based on generated data
    run_hmatrix_solve_cholesky();

    // Calculate relative error
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

    out = inst.eval_string("product_hmat_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("x_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE("H-matrix solve equations by Cholesky factorization in situ",
          "[hmatrix]")
{
  // Create a unique working directory for each test case
  volatile HBEMTestScopedDirectory scoped_dir;

  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  auto trial_no = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    // Predefine Octave and C++ random seed to make tests repeatable
    int rng_seed = 1234567 + trial_no * 7;
    // TODO `src/aca_plus.cu` and `include/aca_plus.hcu` use a
    // std::random_device for hardware seeding, need a mechansim to
    // set the seed.

    std::ostringstream oss;
    oss << "rand('seed'," << rng_seed << ");\n";
    oss << "randn('seed'," << rng_seed << ");";
    inst.eval_string(oss.str());

    // Execute script `gen_matrix.m` to generate M.dat and b.dat
    REQUIRE_NOTHROW([&]() { inst.source_file(SOURCE_DIR "/gen_matrix.m"); }());

    // Run solving based on generated data
    run_hmatrix_solve_cholesky_in_situ();

    // Calculate relative error
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

    out = inst.eval_string("product_hmat_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));

    out = inst.eval_string("x_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
  }
}
