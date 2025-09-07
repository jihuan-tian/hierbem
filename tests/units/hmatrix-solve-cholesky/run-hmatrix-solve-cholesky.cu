#include <catch2/catch_all.hpp>

#include <fstream>

#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/cu_debug_tools.hcu"
#include "utilities/read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_solve_cholesky()
{
  std::ofstream ofs("hmatrix-solve-cholesky.output");

  /**
   * Read a full matrix where only the lower triangular part (including the
   * diagonal) is stored.
   */
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();
  REQUIRE(M.size()[0] > 0);
  REQUIRE(M.size()[0] == M.size()[1]);

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
   * Create the \hmatrix from the source full matrix @p M, where only the lower
   * triangular part is stored.
   *
   * N.B. Because the full matrix has been assigned the @p symmetric property,
   * the created \hmatrix will be automatically set to @p symmetric, which is
   * mandatory for the following Cholesky factorization.
   */
  const unsigned int fixed_rank = n / 4;
  HMatrix<3, double> H(bct, M, fixed_rank, HMatrixSupport::diagonal_block);
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

  /**
   * Create the \hmatrix storing the result of Cholesky factorization, where
   * only the lower triangular part is effective. The property of this matrix
   * should be set to @p lower_triangular.
   */
  HMatrix<3, double> L(bct,
                       fixed_rank,
                       HMatrixSupport::lower_triangular,
                       HMatrixSupport::diagonal_block);

  /**
   * Perform the Cholesky factorization.
   */
  H.compute_cholesky_factorization(L, fixed_rank);

  ofs << "H's state after Cholesky factorization: "
      << HMatrixSupport::state_name(H.get_state()) << "\n";
  ofs << "H's property after Cholesky factorization: "
      << HMatrixSupport::property_name(H.get_property()) << "\n";

  ofs << "L's state after Cholesky factorization: "
      << HMatrixSupport::state_name(L.get_state()) << "\n";
  ofs << "L's property after Cholesky factorization: "
      << HMatrixSupport::property_name(L.get_property()) << std::endl;

  /**
   * Recalculate the rank values (upper bound only) for all rank-k matrices in
   * the resulted \hmatrix \p L, which will be used in the plot of its
   * \hmatrix structure.
   */
  L.calc_rank_upper_bound_for_rkmatrices();

  /**
   * Print the \bct structure of the Cholesky \hmatrix.
   */
  std::ofstream L_bct("L_bct.dat");
  L.write_leaf_set_by_iteration(L_bct);
  L_bct.close();

  /**
   * Convert the \Hcal-Cholesky factor to full matrix.
   */
  LAPACKFullMatrixExt<double> L_full;
  L.convertToFullMatrix(L_full);
  REQUIRE(L_full.size()[0] == M.size()[0]);
  REQUIRE(L_full.size()[1] == M.size()[1]);

  ofs << "L_full's property: "
      << LAPACKSupport::property_name(L_full.get_property()) << std::endl;
  L_full.print_formatted_to_mat(ofs, "L_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  L.solve_cholesky(x, b);
  REQUIRE(x.size() == b.size());

  /**
   * Print the result vector.
   */
  print_vector_to_mat(ofs, "x", x);

  ofs.close();
}


void
run_hmatrix_solve_cholesky_in_situ()
{
  std::ofstream ofs("hmatrix-solve-cholesky.output");

  /**
   * Read a full matrix where only the lower triangular part (including the
   * diagonal) is stored.
   */
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();
  REQUIRE(M.size()[0] > 0);
  REQUIRE(M.size()[0] == M.size()[1]);

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
   * Create the \hmatrix from the source full matrix @p M, where only the lower
   * triangular part is stored.
   *
   * N.B. Because the full matrix has been assigned the @p symmetric property,
   * the created \hmatrix will be automatically set to @p symmetric, which is
   * mandatory for the following Cholesky factorization.
   */
  const unsigned int fixed_rank = n / 4;
  HMatrix<3, double> H(bct, M, fixed_rank, HMatrixSupport::diagonal_block);
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

  /**
   * Perform the Cholesky factorization.
   */
  H.compute_cholesky_factorization(fixed_rank);

  ofs << "H's state after Cholesky factorization: "
      << HMatrixSupport::state_name(H.get_state()) << "\n";
  ofs << "H's property after Cholesky factorization: "
      << HMatrixSupport::property_name(H.get_property()) << "\n";

  /**
   * Recalculate the rank values (upper bound only) for all rank-k matrices in
   * the resulted \hmatrix \p H, which will be used in the plot of its
   * \hmatrix structure.
   */
  H.calc_rank_upper_bound_for_rkmatrices();

  /**
   * Print the \bct structure of the Cholesky \hmatrix.
   */
  std::ofstream L_bct("L_bct.dat");
  H.write_leaf_set_by_iteration(L_bct);
  L_bct.close();

  /**
   * Convert the \Hcal-Cholesky factor to full matrix.
   */
  LAPACKFullMatrixExt<double> L_full;
  H.convertToFullMatrix(L_full);
  REQUIRE(L_full.size()[0] == M.size()[0]);
  REQUIRE(L_full.size()[1] == M.size()[1]);

  ofs << "L_full's property: "
      << LAPACKSupport::property_name(L_full.get_property()) << std::endl;
  L_full.print_formatted_to_mat(ofs, "L_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  H.solve_cholesky(x, b);
  REQUIRE(x.size() == b.size());

  /**
   * Print the result vector.
   */
  print_vector_to_mat(ofs, "x", x);

  ofs.close();
}
