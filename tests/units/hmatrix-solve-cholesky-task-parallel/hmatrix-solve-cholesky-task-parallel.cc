/**
 * \file hmatrix-solve-cholesky-task-parallel.cc
 * \brief Verify Cholesky parallel factorization of a positive definite and
 * symmetric \hmatrix and solve this matrix using forward and backward
 * substitution.
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
 * \date 2024-01-06
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
run_hmatrix_solve_cholesky_task_parallel(const unsigned int trial_no);

static constexpr int FUZZING_TIMES = 5;
const unsigned int   REPEAT_TIMES  = 100;

TEST_CASE("H-matrix solve equations by parallel Cholesky factorization",
          "[hmatrix]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(HBEM_ROOT_DIR "/scripts");
  inst.add_path(SOURCE_DIR);

  auto trial_no = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    // Predefine Octave and C++ random seed to make tests repeatable
    int rng_seed = 1543267 + trial_no * 7;
    // TODO `src/aca_plus.cu` and `include/aca_plus.hcu` use a
    // std::random_device for hardware seeding, need a mechansim to
    // set the seed.

    std::ostringstream oss;
    oss << "rand('seed'," << rng_seed << ");\n";
    oss << "randn('seed'," << rng_seed << ");";
    inst.eval_string(oss.str());

    // Execute script `gen_matrix.m` to generate M.dat and b.dat
    REQUIRE_NOTHROW([&]() {
      inst.eval_function_void(std::string("gen_matrix(") +
                              std::to_string(trial_no) + std::string(");"));
    }());

    for (unsigned int i = 0; i < REPEAT_TIMES; i++)
      {
        // Run solving based on generated data
        run_hmatrix_solve_cholesky_task_parallel(trial_no);

        // Calculate relative error
        try
          {
            inst.eval_string(
              std::string(
                "[MM_cond, hmat_rel_err, product_hmat_rel_err, x_octave, x_rel_err, hmat_factorized_rel_err] = process(") +
              std::to_string(trial_no) + std::string(")"));
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

        // Make the solution vector x's error threshold depend on the condition
        // number of the system matrix.
        out = inst.eval_string("MM_cond");

        double x_rel_err_threshold;
        if (out.double_value() > 1e6)
          // When the condition number is large, relax the threshold.
          x_rel_err_threshold = 1e-5;
        else
          x_rel_err_threshold = 1e-6;

        out = inst.eval_string("x_rel_err");
        REQUIRE_THAT(out.double_value(), WithinAbs(0.0, x_rel_err_threshold));

        out = inst.eval_string("hmat_factorized_rel_err");
        REQUIRE_THAT(out.double_value(), WithinAbs(0.0, 1e-10));
      }
  }
}
