/**
 * \file hmatrix-solve-cholesky-in-situ-for-erichsen1996efficient-example2.cc
 * \brief Verify in situ Cholesky factorization of a positive definite and
 * symmetric (SPD) \hmatrix and solve this matrix using forward and backward
 * substitution. This \hmatrix is generated from a discretization of Example 2
 * in Erichsen1996Efficient paper.
 *
 * For testing purpose, run with the following arguments
 *
 * 1. Perform SLP full matrix computation using the Sauter quadrature and output
 * the full matrix.
 *
 * <code>
 * ./progname gen_input_matrices
 * </code>
 *
 * 2. Perform H-matrix construction by reading the SLP full matrix data.
 * Truncation of the \hmatrix is carried out directly.
 *
 * <code>
 * ./progname build_hmat [truncation_rank] normal
 * </code>
 *
 * 3. Perform H-matrix construction by reading the SLP full matrix data.
 * Truncation of the \hmatrix is carried out preserving positive definiteness.
 *
 * <code>
 * ./progname build_hmat [truncation_rank] spd
 * </code>
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-12-03
 */

#include <fstream>
#include <iostream>

#include "debug_tools.h"
#include "erichsen1996efficient_example2.h"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

int
main(int argc, char *argv[])
{
  (void)argc;

  /**
   * Create the test case object for Example 2 in Erichsen1996Efficient.
   */
  const unsigned int                       fe_order = 1;
  IdeoBEM::Erichsen1996Efficient::Example2 testcase("sphere-from-gmsh_hex.msh",
                                                    fe_order);

  std::string run_type(argv[1]);
  if (run_type == std::string("gen_input_matrices"))
    {
      testcase.build_slp_only(true);

      /**
       * Calculate the system matrices.
       */
      LAPACKFullMatrix<double> system_rhs_slp_matrix(
        testcase.get_system_rhs_matrix().m(),
        testcase.get_system_rhs_matrix().n());
      system_rhs_slp_matrix = testcase.get_system_rhs_matrix();
      LAPACKFullMatrixExt<double> system_rhs_slp_matrix_ext(
        system_rhs_slp_matrix);
      system_rhs_slp_matrix_ext.print_formatted_to_mat(
        std::cout, "M", 15, false, 25, "0");
      Vector<double> &system_rhs = testcase.get_system_rhs();
      print_vector_to_mat(std::cout, "b", system_rhs, false);

      return 0;
    }
  else if (run_type == std::string("build_hmat"))
    {
      testcase.build_slp_only(false);

      /**
       * Read the system matrices from previous saved data.
       *
       * \mycomment{2022-04-13: Even though the system matrices are symmetric,
       * all of their elements have been calculated due to the unoptimized
       * implementation up to now.}
       */
      LAPACKFullMatrixExt<double> system_rhs_slp_matrix_ext;
      std::ifstream               in("input_matrices.dat");
      read_matrix_from_octave(in, "M", system_rhs_slp_matrix_ext);
      system_rhs_slp_matrix_ext.set_property(LAPACKSupport::symmetric);

      Vector<double> system_rhs;
      in.close();
      in.open("input_matrices.dat");
      read_vector_from_octave(in, "b", system_rhs);

      BlockClusterTree<3> &bct = testcase.get_bct();

      const unsigned int fixed_rank = std::atoi(argv[2]);
      /**
       * Convert the SLP full matrix to an \hmatrix.
       *
       * \mycomment{2022-04-13: Even though the system matrix SLP here is
       * symmetric, all of its elements have been calculated due to the
       * unoptimized implementation up to now. Meanwhile, the conversion from
       * this full matrix to \hmatrix applies to all matrix blocks instead of
       * only the diagonal and lower triangular blocks.
       *
       * 2022-05-10: Now the symmetric \hmatrix is implemented and only the
       * lower triangular part and the diagonal part are converted.}
       */
      HMatrix<3, double> H(bct,
                           system_rhs_slp_matrix_ext,
                           HMatrixSupport::diagonal_block);

      std::string truncation_method(argv[3]);
      if (truncation_method == std::string("normal"))
        {
          /**
           * Perform a normal \hmatrix rank truncation without caring about its
           * positive definiteness.
           */
          H.truncate_to_rank(fixed_rank);
        }
      else if (truncation_method == std::string("spd"))
        {
          /**
           * Perform rank truncation to both lower and upper triangular parts of
           * the matrix.
           */
          H.truncate_to_rank_preserve_positive_definite(fixed_rank, false);
        }
      else
        {
          Assert(false, ExcInternalError());
        }

      /**
       * Convert the \hmatrix to a full matrix and write it to a file for
       * verification.
       */
      LAPACKFullMatrixExt<double> H_full;
      H.convertToFullMatrix(H_full);
      H_full.print_formatted_to_mat(std::cout, "H_full", 15, false, 25, "0");

      std::ofstream H_bct("H_bct.dat");
      H.write_leaf_set_by_iteration(H_bct);
      H_bct.close();

      /**
       * Perform Cholesky factorization on the \hmatrix. Even though the
       * \hmatrix holds both lower and upper triangular blocks, the Cholesky
       * factorization will only handle the diagonal and lower triangular
       * blocks.
       */
      H.compute_cholesky_factorization(fixed_rank);

      /**
       * Recalculate the rank values (upper bound only) for all rank-k matrices
       * in the resulted \hmatrix.
       */
      H.calc_rank_upper_bound_for_rkmatrices();

      /**
       * Print the \bct structure of the Cholesky \hmatrix.
       */
      H_bct.open("LLT_bct.dat");
      H.write_leaf_set_by_iteration(H_bct);
      H_bct.close();

      /**
       * Convert the \Hcal-Cholesky factor to full matrix.
       */
      LAPACKFullMatrixExt<double> LLT_full;
      H.convertToFullMatrix(LLT_full);
      LLT_full.print_formatted_to_mat(
        std::cout, "LLT_full", 15, false, 25, "0");

      /**
       * Solve the matrix.
       */
      Vector<double> x;
      H.solve_cholesky(x, system_rhs);

      /**
       * Print the result vector.
       */
      print_vector_to_mat(std::cout, "x", x);

      return 0;
    }
  else
    {
      Assert(false, ExcInternalError());
    }
}
