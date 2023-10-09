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
#include <octave/interpreter.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

#include <fstream>
#include <iostream>

#include "debug_tools.hcu"
#include "hbem_test_config.h"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

#define TOTAL_FUZZING 5
#define STR(x) #x
#define XSTR(x) STR(x)

using namespace HierBEM;
using namespace Catch::Matchers;

int
run_hmatrix_solve_lu()
{
  std::ofstream ofs("hmatrix-solve-lu.output");

  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

  Vector<double> b;
  in.open("b.dat");
  read_vector_from_octave(in, "b", b);
  in.close();

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

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
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
  LU_full.print_formatted_to_mat(ofs, "LU_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  LU.solve_lu(x, b);

  /**
   * Print the result vector.
   */
  print_vector_to_mat(ofs, "x", x);

  return 0;
}

TEST_CASE("H-matrix solve equations by LU decomposition (fuzz " XSTR(
            TOTAL_FUZZING) " times)",
          "[hmatrix]")
{
  // Create Octave interpreter
  octave::interpreter interpreter;

  // Initialize and start Octave interpreter instance
  int status = interpreter.execute();
  REQUIRE(status == 0);

  // Add additional Octave search path
  octave_value_list in;
  in(0) = HBEM_ROOT_DIR "/scripts";
  Faddpath(interpreter, in);

  for (int fuzz_no = 0; fuzz_no < TOTAL_FUZZING; fuzz_no++)
    {
      std::cout << "Fuzz No. " << fuzz_no << std::endl;

      // Execute Octave script to generate M.dat and b.dat
      REQUIRE_NOTHROW([&]() {
        // This may throw exception
        octave::source_file(SOURCE_DIR "/gen_matrix.m");
      }());

      REQUIRE_NOTHROW(run_hmatrix_solve_lu());

      // Execute Octave script to verify result and generate plots
      int parse_status;
      if (fuzz_no == TOTAL_FUZZING - 1)
        {
          // The last run, generate plots
          interpreter.eval_string("enable_figure=1", true, parse_status);
          REQUIRE(parse_status == 0);
        }
      else
        {
          // Middle runs, do not generate plots
          interpreter.eval_string("enable_figure=0", true, parse_status);
          REQUIRE(parse_status == 0);
        }

      REQUIRE_NOTHROW([&]() {
        // This may throw exception
        octave::source_file(SOURCE_DIR "/process.m");
      }());

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
