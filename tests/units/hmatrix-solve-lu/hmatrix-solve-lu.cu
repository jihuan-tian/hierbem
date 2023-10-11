/**
 * \file hmatrix-solve-lu.cu
 * \brief Verify LU factorization of an \hmatrix and solve this matrix using
 * forward and backward substitution.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-02
 */
#include <catch2/catch_all.hpp>
#include <octave/builtin-defun-decls.h>
// XXX UGLY TRICK: `octave/graphics.h` and `cuda_surface_types.h` both defined
// global class `surface`, the former one will be included by
// `octave/interpreter.h` and the latter one **WILL ALWAYS** be included when
// using CUDA nvcc compiler. Therefor the naming confliction on `surface` seems
// to be unavoidable. As a workaround, we have to define the protecting macro
// `octave/graphics_h` before including `octave/interpreter.h` to prevent
// the expansion of the former.
#define octave_graphics_h 1
#include <octave/interpreter.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "debug_tools.hcu"
#include "hbem_test_config.h"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_solve_lu()
{
  std::ofstream ofs("hmatrix-solve-lu.output");

  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();
  REQUIRE(M.size()[0] > 0);
  REQUIRE(M.size()[0] == M.size()[1]);

  Vector<double> b;
  in.open("b.dat");
  read_vector_from_octave(in, "b", b);
  in.close();
  REQUIRE(b.size() == M.size()[0]);

  /**
   * Generate index set.
   */
  const unsigned int                   n = 32;
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min = 2;

  /**
   * Generate cluster tree.
   */
  ClusterTree<3, double> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree with the two component cluster trees being the
   * same.
   */
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree);
  bct.partition_fine_non_tensor_product();

  /**
   * Create the \hmatrix.
   */
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, M, fixed_rank);
  REQUIRE(H.get_m() == M.size()[0]);
  REQUIRE(H.get_n() == M.size()[1]);

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  REQUIRE(H_full.size()[0] == M.size()[0]);
  REQUIRE(H_full.size()[1] == M.size()[1]);

  H_full.print_formatted_to_mat(ofs, "H_full", 15, false, 25, "0");

  HMatrix<3, double> LU(bct, fixed_rank);

  /**
   * Perform LU factorization.
   */
  H.compute_lu_factorization(LU, fixed_rank);
  ofs << "H's state after LU factorization: "
      << HMatrixSupport::state_name(H.get_state()) << "\n";
  ofs << "LU's state after LU factorization: "
      << HMatrixSupport::state_name(LU.get_state()) << std::endl;

  /**
   * Print the \bct structure of the LU \hmatrix.
   */
  std::ofstream LU_bct("LU_bct.dat");
  LU.write_leaf_set_by_iteration(LU_bct);
  LU_bct.close();

  /**
   * Convert the \Hcal-LU matrix to full matrix.
   */
  LAPACKFullMatrixExt<double> LU_full;
  LU.convertToFullMatrix(LU_full);
  REQUIRE(LU_full.size()[0] == M.size()[0]);
  REQUIRE(LU_full.size()[1] == M.size()[1]);

  LU_full.print_formatted_to_mat(ofs, "LU_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  LU.solve_lu(x, b);
  REQUIRE(x.size() == b.size());

  /**
   * Print the result vector.
   */
  print_vector_to_mat(ofs, "x", x);
}

static constexpr int FUZZING_TIMES = 5;
TEST_CASE("H-matrix solve equations by LU decomposition", "[hmatrix]")
{
  // Catch2 generator will run the code multiple times in the same thread,
  // so the Octave interpreter must be declared STATIC to ensure there is
  // only one active instance per thread!
  static octave::interpreter interpreter;

  // Initialize and start Octave interpreter instance
  int status = interpreter.execute();
  REQUIRE(status == 0);

  // Add additional Octave search path to include Octave utilities needed
  octave_value_list in;
  in(0) = HBEM_ROOT_DIR "/scripts";
  Faddpath(interpreter, in);

  auto trial_no = GENERATE(range(0, FUZZING_TIMES));
  SECTION(std::string("trial #") + std::to_string(trial_no))
  {
    int parse_status;

    // Predefine Octave and C++ random seed to make tests repeatable
    int rng_seed = 1234567 + trial_no * 7;
    // TODO `src/aca_plus.cu` and `include/aca_plus.hcu` use a
    // std::random_device for hardware seeding, need a mechansim to
    // set the seed.

    std::ostringstream oss;
    oss << "rand('seed'," << rng_seed << ");\n";
    oss << "randn('seed'," << rng_seed << ");";
    interpreter.eval_string(oss.str(), true, parse_status);
    REQUIRE(parse_status == 0);

    // Execute script `gen_matrix.m` to generate M.dat and b.dat
    REQUIRE_NOTHROW([&]() {
      // This may throw exception
      octave::source_file(SOURCE_DIR "/gen_matrix.m");
    }());

    // Run solving based on generated data
    run_hmatrix_solve_lu();

    // Execute Octave script to verify result and generate plots
    int enable_figure =
      (trial_no == FUZZING_TIMES - 1) ? 1 : 0; // Generate plots on the last run

    // Set enable_figure Octave variable
    interpreter.eval_string(std::string("enable_figure=") +
                              std::to_string(enable_figure),
                            true,
                            parse_status);
    REQUIRE(parse_status == 0);

    // Calculate relative error and draw plots if enable_figure==1
    REQUIRE_NOTHROW([&]() {
      // This may throw exception
      octave::source_file(SOURCE_DIR "/process.m");
    }());

    // Check relative error
    octave_value out;
    out = interpreter.eval_string("hmat_rel_err", true, parse_status);
    REQUIRE(parse_status == 0);
    REQUIRE_THAT(out.double_value(),
                 WithinAbs(0.0, 1e-6) || WithinRel(0.0, 1e-8));

    out = interpreter.eval_string("x_rel_err", true, parse_status);
    REQUIRE(parse_status == 0);
    REQUIRE_THAT(out.double_value(),
                 WithinAbs(0.0, 1e-6) || WithinRel(0.0, 1e-8));
  }
}
