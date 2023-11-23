#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>

#include "debug_tools.hcu"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_forward_substitution_matrix_valued()
{
  std::ofstream ofs("hmatrix-forward-substitution-matrix-valued.output");

  LAPACKFullMatrixExt<double> L;
  std::ifstream               in("L.dat");
  L.read_from_mat(in, "L");
  in.close();
  REQUIRE(L.size()[0] > 0);
  REQUIRE(L.size()[0] == L.size()[1]);

  LAPACKFullMatrixExt<double> Z;
  in.open("Z.dat");
  Z.read_from_mat(in, "Z");
  in.close();
  REQUIRE(Z.size()[0] > 0);
  REQUIRE(Z.size()[1] > 0);

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
   * Generate block cluster tree for \p X with the two component cluster trees
   * being the same.
   */
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree, n_min);
  bct.partition_fine_non_tensor_product();

  /**
   * Generate block cluster tree for \p Z with the two component cluster trees
   * being the same. N.B. The @p n_min value here is intentionally set to a
   * different value.
   */
  BlockClusterTree<3, double> bct_rhs(cluster_tree, cluster_tree, n_min * 2);
  bct_rhs.partition_fine_non_tensor_product();

  /**
   * Create \hmatrices. Here we enforce each far field matrix block to have a
   * full rank, i.e. its rank is the same as that of the corresponding full
   * matrix block. In this way, the constructed H-matrix can accurately
   * represent the original full matrix, which produces a very small relative
   * error in the following computation and make catch2 tests pass.
   *
   * For an H-matrix having the fine non-tensor product partition, the largest
   * far field matrix block size is @p n/4. Therefore, we use this value as the
   * fixed matrix rank.
   */
  const unsigned int fixed_rank = n / 4;
  HMatrix<3, double> HL(bct, L, fixed_rank);
  std::ofstream      out("HL_bct.dat");
  HL.write_leaf_set_by_iteration(out);
  out.close();

  LAPACKFullMatrixExt<double> L_full;
  HL.convertToFullMatrix(L_full);
  REQUIRE(L_full.size()[0] == L.size()[0]);
  REQUIRE(L_full.size()[1] == L.size()[1]);

  L_full.print_formatted_to_mat(ofs, "L_full", 15, false, 25, "0");

  HMatrix<3, double> HZ(bct_rhs, Z, fixed_rank);
  out.open("HZ_bct.dat");
  HZ.write_leaf_set_by_iteration(out);
  out.close();

  LAPACKFullMatrixExt<double> Z_full;
  HZ.convertToFullMatrix(Z_full);
  REQUIRE(Z_full.size()[0] == Z.size()[0]);
  REQUIRE(Z_full.size()[1] == Z.size()[1]);

  Z_full.print_formatted_to_mat(ofs, "Z_full", 15, false, 25, "0");

  /**
   * Create the empty solution \hmatrix \p X with memory allocated but without
   * data.
   */
  HMatrix<3, double> HX(bct_rhs, fixed_rank);

  /**
   * Solve the matrix using matrix-valued forward substitution.
   */
  HL.solve_by_forward_substitution_matrix_valued(HX, HZ, fixed_rank, false);
  out.open("HX_bct.dat");
  HX.write_leaf_set_by_iteration(out);
  out.close();

  /**
   * Convert the result \hmatrix \p X to a full matrix.
   */
  LAPACKFullMatrixExt<double> X;
  HX.convertToFullMatrix(X);
  REQUIRE(X.size()[0] == HX.get_m());
  REQUIRE(X.size()[1] == HX.get_n());

  X.print_formatted_to_mat(ofs, "X", 15, false, 25, "0");

  ofs.close();
}

void
run_hmatrix_forward_substitution_matrix_valued_in_situ()
{
  std::ofstream ofs("hmatrix-forward-substitution-matrix-valued.output");

  LAPACKFullMatrixExt<double> L;
  std::ifstream               in("L.dat");
  L.read_from_mat(in, "L");
  in.close();
  REQUIRE(L.size()[0] > 0);
  REQUIRE(L.size()[0] == L.size()[1]);

  LAPACKFullMatrixExt<double> Z;
  in.open("Z.dat");
  Z.read_from_mat(in, "Z");
  in.close();
  REQUIRE(Z.size()[0] > 0);
  REQUIRE(Z.size()[1] > 0);

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
   * Generate block cluster tree for \p X with the two component cluster trees
   * being the same.
   */
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree, n_min);
  bct.partition_fine_non_tensor_product();

  /**
   * Generate block cluster tree for \p Z with the two component cluster trees
   * being the same. N.B. The @p n_min value here is intentionally set to a
   * different value.
   */
  BlockClusterTree<3, double> bct_rhs(cluster_tree, cluster_tree, n_min * 2);
  bct_rhs.partition_fine_non_tensor_product();

  /**
   * Create \hmatrices. Here we enforce each far field matrix block to have a
   * full rank, i.e. its rank is the same as that of the corresponding full
   * matrix block. In this way, the constructed H-matrix can accurately
   * represent the original full matrix, which produces a very small relative
   * error in the following computation and make catch2 tests pass.
   *
   * For an H-matrix having the fine non-tensor product partition, the largest
   * far field matrix block size is @p n/4. Therefore, we use this value as the
   * fixed matrix rank.
   */
  const unsigned int fixed_rank = n / 4;
  HMatrix<3, double> HL(bct, L, fixed_rank);
  std::ofstream      out("HL_bct.dat");
  HL.write_leaf_set_by_iteration(out);
  out.close();

  LAPACKFullMatrixExt<double> L_full;
  HL.convertToFullMatrix(L_full);
  REQUIRE(L_full.size()[0] == L.size()[0]);
  REQUIRE(L_full.size()[1] == L.size()[1]);

  L_full.print_formatted_to_mat(ofs, "L_full", 15, false, 25, "0");

  HMatrix<3, double> HZ(bct_rhs, Z, fixed_rank);
  out.open("HZ_bct.dat");
  HZ.write_leaf_set_by_iteration(out);
  out.close();

  LAPACKFullMatrixExt<double> Z_full;
  HZ.convertToFullMatrix(Z_full);
  REQUIRE(Z_full.size()[0] == Z.size()[0]);
  REQUIRE(Z_full.size()[1] == Z.size()[1]);

  Z_full.print_formatted_to_mat(ofs, "Z_full", 15, false, 25, "0");

  /**
   * Solve the matrix using matrix-valued forward substitution. The result
   * directly overwrites the right hand side matrix @p Z.
   */
  HL.solve_by_forward_substitution_matrix_valued(HZ, fixed_rank, false);

  /**
   * Convert the result \hmatrix \p Z to a full matrix.
   */
  LAPACKFullMatrixExt<double> X;
  HZ.convertToFullMatrix(X);
  REQUIRE(X.size()[0] == HZ.get_m());
  REQUIRE(X.size()[1] == HZ.get_n());

  X.print_formatted_to_mat(ofs, "X", 15, false, 25, "0");

  ofs.close();
}
