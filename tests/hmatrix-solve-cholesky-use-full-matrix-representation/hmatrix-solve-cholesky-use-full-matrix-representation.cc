/**
 * \file hmatrix-solve-cholesky-use-full-matrix-representation.cc
 * \brief Verify Cholesky factorization of a positive definite and symmetric
 * \hmatrix and solve this matrix using forward and backward substitution.
 *
 * \details In this tester, the property of the \hmatrix is @p general instead
 * of @p symmetric.
 *
 * \alert{If there is no special treatment as that proposed by Bebendorf, the
 * approximation of the original full matrix using \hmatrix must be good enough
 * so that the positive definiteness of the original matrix is preserved and
 * Cholesky factorization is applicable.}
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-05-06
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
   * diagonal) is stored. But the matrix property is still set to @p general.
   */
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

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
   * Create the \hmatrix from the full matrix @p M with the property @p general.
   * N.B. This is just for test purpose.
   */
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, M, fixed_rank, HMatrixSupport::diagonal_block);
  /**
   * Even though the memory has been allocated to all blocks in the \hmatrix,
   * here we set its property to @p symmetric, to which the Cholesky
   * factorization can be applied.
   */
  H.set_property(HMatrixSupport::symmetric);

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 15, false, 25, "0");

  /**
   * Create the \hmatrix storing the result of Cholesky factorization. Here,
   * its property is set to @p general just for test purpose, so that the memory
   * will be allocated for all of its blocks.
   */
  HMatrix<3, double> LLT(bct,
                         fixed_rank,
                         HMatrixSupport::general,
                         HMatrixSupport::diagonal_block);
  /**
   * Even though the memory has been allocated to all blocks in the result
   * \hmatrix, here we set its property to @p lower_triangular, which is
   * required by the Cholesky factorization to be performed.
   */
  LLT.set_property(HMatrixSupport::lower_triangular);
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
  LLT_full.print_formatted_to_mat(std::cout, "LLT_full", 15, false, 25, "0");

  /**
   * Solve the matrix. \alert{Before solving the matrix, the matrix state
   * should be set to @p cholesky explicitly, which is required by the function
   * @p solve_cholesky.}
   */
  Vector<double> x;
  LLT.solve_cholesky(x, b);

  /**
   * Print the result vector.
   */
  print_vector_to_mat(std::cout, "x", x);
}
