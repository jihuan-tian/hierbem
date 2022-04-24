/**
 * \file hmatrix-solve-cholesky.cc
 * \brief Verify Cholesky factorization of a positive definite and symmetric
 * \hmatrix and solve this matrix using forward and backward substitution.
 *
 * \details \alert{If there is no special treatment as that proposed by
 * Bebendorf, the approximation of the original full matrix using \hmatrix must
 * be good enough so that the positive definiteness of the original matrix is
 * preserved and Cholesky factorization is applicable.}
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
   * Create the \hmatrix from the source matrix, where only the lower triangular
   * part is stored. N.B. Because the full matrix has been assigned the
   * @p symmetric property, the created \hmatrix will be automatically set to
   * @p symmetric, which is mandatory for the following Cholesky factorization.
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
   * Create the \hmatrix after Cholesky factorization, where only the lower
   * triangular part is effective. The property of this matrix should be set to
   * @p symmetric.
   */
  HMatrix<3, double> LLT(bct,
                         fixed_rank,
                         HMatrixSupport::symmetric,
                         HMatrixSupport::diagonal_block);

  /**
   * Perform Cholesky factorization. After that, the whole \hmatrix hierarchy
   * should be set to the @p cholesky state.
   */
  H.compute_cholesky_factorization(LLT, fixed_rank);
  /**
   * After performing Cholesky factorization, the state of the original
   * \hmatrix should be set to @p unusable and the state of the result \hmatrix
   * should be set to @p cholesky.
   */
  H.set_state(HMatrixSupport::unusable);
  LLT.set_state(HMatrixSupport::cholesky);
  /**
   * Recalculate the rank values (upper bound only) for all rank-k matrices in
   * the resulted \hmatrix \p LLT.
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
   * Solve the matrix.
   */
  Vector<double> x;
  LLT.solve_cholesky(x, b);

  /**
   * Print the result vector.
   */
  print_vector_to_mat(std::cout, "x", x);
}
