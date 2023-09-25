/**
 * \file hmatrix-solve-cholesky-in-situ-use-full-matrix-representation.cc
 * \brief Verify in situ Cholesky factorization of a positive definite and
 * symmetric \hmatrix and solve this matrix using forward and backward
 * substitution.
 *
 * In this tester, the property of the \hmatrix is @p general instead of
 * @p symmetric. \alert{If there is no special treatment as that proposed by
 * Bebendorf, the approximation of the original full matrix using \hmatrix must
 * be good enough so that the positive definiteness of the original matrix is
 * preserved and Cholesky factorization is applicable.}
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-05-10
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
  // Read a full matrix where only the lower triangular part (including the
  // diagonal) is stored.
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
   * Create the \hmatrix from the source full matrix @p M, where only the lower
   * triangular part is stored.
   *
   * N.B. Because the full matrix has not been assigned the @p symmetric
   * property, the created \hmatrix will have memory allocated for all of its
   * blocks. However, after the \hmatrix is created, its property should be
   * manually set to @p symmetric, which is required by the internal assertions
   * in @p HMatrix::_compute_cholesky_factorization.
   */
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, M, HMatrixSupport::diagonal_block);

  std::cout << "H's state before Cholesky factorization: "
            << HMatrixSupport::state_name(H.get_state()) << "\n";
  std::cout << "H's property before Cholesky factorization: "
            << HMatrixSupport::property_name(H.get_property()) << "\n";

  H.set_property(HMatrixSupport::symmetric);

  // H.truncate_to_rank(fixed_rank);
  H.truncate_to_rank_preserve_positive_definite(fixed_rank);

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 15, false, 25, "0");

  /**
   * Perform Cholesky factorization.
   */
  H.compute_cholesky_factorization(fixed_rank);

  std::cout << "H's state after Cholesky factorization: "
            << HMatrixSupport::state_name(H.get_state()) << "\n";
  std::cout << "H's property after Cholesky factorization: "
            << HMatrixSupport::property_name(H.get_property()) << "\n";

  /**
   * Recalculate the rank values (upper bound only) for all rank-k matrices in
   * the resulted \hmatrix.
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
  LLT_full.print_formatted_to_mat(std::cout, "LLT_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  H.solve_cholesky(x, b);

  /**
   * Print the result vector.
   */
  print_vector_to_mat(std::cout, "x", x);
}
