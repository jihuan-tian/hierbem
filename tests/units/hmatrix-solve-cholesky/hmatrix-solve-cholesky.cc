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
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-13
 */

#include <fstream>
#include <iostream>

#include "debug_tools.h"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

int
main()
{
  /**
   * Read a full matrix where only the lower triangular part (including the
   * diagonal) is stored.
   */
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

  /**
   * Set the property of the full matrix as @p symmetric.
   */
  M.set_property(LAPACKSupport::symmetric);

  /**
   * Read the RHS vector.
   */
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
   * Create the \hmatrix from the source full matrix @p M, where only the lower
   * triangular part is stored.
   *
   * N.B. Because the full matrix has been assigned the @p symmetric property,
   * the created \hmatrix will be automatically set to @p symmetric, which is
   * mandatory for the following Cholesky factorization.
   */
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, M, fixed_rank, HMatrixSupport::diagonal_block);

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 15, false, 25, "0");

  /**
   * Create the \hmatrix storing the result of Cholesky factorization, where
   * only the lower triangular part is effective. The property of this matrix
   * should be set to
   * @p lower_triangular.
   */
  HMatrix<3, double> LLT(bct,
                         fixed_rank,
                         HMatrixSupport::lower_triangular,
                         HMatrixSupport::diagonal_block);
  std::cout << "LLT memory consumption before Cholesky factorization: "
            << LLT.memory_consumption() << std::endl;

  /**
   * Perform the Cholesky factorization.
   */
  H.compute_cholesky_factorization(LLT, fixed_rank);

  std::cout << "H's state after Cholesky factorization: "
            << HMatrixSupport::state_name(H.get_state()) << "\n";
  std::cout << "H's property after Cholesky factorization: "
            << HMatrixSupport::property_name(H.get_property()) << "\n";

  std::cout << "LLT's state after Cholesky factorization: "
            << HMatrixSupport::state_name(LLT.get_state()) << "\n";
  std::cout << "LLT's property after Cholesky factorization: "
            << HMatrixSupport::property_name(LLT.get_property()) << std::endl;
  std::cout << "LLT memory consumption: " << LLT.memory_consumption() << "\n";
  std::cout << "LLT coarse memory consumption: "
            << LLT.memory_consumption_for_core_data() << std::endl;

  /**
   * Recalculate the rank values (upper bound only) for all rank-k matrices in
   * the resulted \hmatrix \p LLT, which will be used in the plot of its
   * \hmatrix structure.
   */
  LLT.calc_rank_upper_bound_for_rkmatrices();

  /**
   * Print the \bct structure of the Cholesky \hmatrix.
   */
  std::ofstream LLT_bct("LLT_bct.dat");
  LLT.write_leaf_set_by_iteration(LLT_bct);
  LLT_bct.close();

  /**
   * Convert the \Hcal-Cholesky factor to full matrix.
   */
  LAPACKFullMatrixExt<double> LLT_full;
  LLT.convertToFullMatrix(LLT_full);
  std::cout << "LLT_full's property: "
            << LAPACKSupport::property_name(LLT_full.get_property())
            << std::endl;
  LLT_full.print_formatted_to_mat(std::cout, "LLT_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  LLT.solve_cholesky(x, b);

  /**
   * Print the result vector.
   */
  print_vector_to_mat(std::cout, "x", x);
}
