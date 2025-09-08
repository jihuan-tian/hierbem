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
 * \file hmatrix-tvmult-tril-serial-iterative.cc
 * \brief Verify serial transposed lower triangular \hmatrix/vector
 * multiplication by iterating over the leaf set.
 *
 * \ingroup test_cases hierarchical_matrices
 * \author Jihuan Tian
 * \date 2024-03-20
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
run_hmatrix_Tvmult_tril_serial_iterative();

static constexpr int FUZZING_TIMES = 50;
TEST_CASE(
  "Transposed lower triangular H-matrix/vector multiplication in serial by iteration",
  "[hmatrix]")
{
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

    run_hmatrix_Tvmult_tril_serial_iterative();

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
    //
    // N.B. The \hmatrix converted from the full matrix uses full rank in low
    // rank matrix blocks, therefore the relative error between the \hmatrix and
    // the original full matrix should be very small.
    //
    // On the other hand, \hmatrix/vector multiplication involves multiple
    // additions from \hmatrix leaf nodes into the result vector, during which
    // there will be round-off errors.
    //
    // Therefore, the error limit for \hmatrix is 1e-14 and that for the result
    // vector from \hmatrix/vector multiplication is loosened a bit, i.e. 1e-12.
    HBEMOctaveValue out;
    out = inst.eval_string("hmat_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("hmat_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y1_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y2_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y3_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y4_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));
  }
}
