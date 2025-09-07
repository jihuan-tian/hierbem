/**
 * \file hmatrix-Hvmult-hermite-symm.cc
 * \brief Verify Hermite transposed \hmatrix/vector multiplication. The \hmatrix
 * is Hermite symmetric and only its lower triangular part is stored.
 *
 * \ingroup test_cases hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-11-28
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
run_hmatrix_Hvmult_hermite_symm();

static constexpr int FUZZING_TIMES = 5;
TEST_CASE("Hermite transposed Hermite symmetric H-matrix/vector multiplication",
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

    // Execute script `gen_matrix.m` to generate M.dat and x.dat
    REQUIRE_NOTHROW([&]() { inst.source_file(SOURCE_DIR "/gen_matrix.m"); }());

    // Run solving based on generated data
    run_hmatrix_Hvmult_hermite_symm();

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
    out = inst.eval_string("hmat_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y1_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y2_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));

    out = inst.eval_string("y3_complex_rel_err");
    REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-14));
  }
}
