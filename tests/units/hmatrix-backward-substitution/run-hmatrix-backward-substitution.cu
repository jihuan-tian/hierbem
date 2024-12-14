#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>

#include "cu_debug_tools.hcu"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_backward_substitution()
{
  std::ofstream ofs("hmatrix-backward-substitution.output");

  LAPACKFullMatrixExt<double> U;
  std::ifstream               in1("U.dat");
  U.read_from_mat(in1, "U");
  in1.close();
  REQUIRE(U.size()[0] > 0);
  REQUIRE(U.size()[0] == U.size()[1]);

  Vector<double> b;
  std::ifstream  in2("b.dat");
  read_vector_from_octave(in2, "b", b);
  in2.close();
  REQUIRE(b.size() > 0);

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
   * Create the \hmatrix. Here we enforce each far field matrix block to have a
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
  HMatrix<3, double> H(bct, U, fixed_rank);
  std::ofstream      H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  REQUIRE(H_full.size()[0] == U.size()[0]);
  REQUIRE(H_full.size()[1] == U.size()[1]);

  H_full.print_formatted_to_mat(ofs, "H_full", 15, false, 25, "0");

  /**
   * Solve the matrix using backward substitution.
   */
  H.solve_by_backward_substitution(b, false);

  /**
   * Print the result vector which has overwritten \p b.
   */
  print_vector_to_mat(ofs, "x", b);
}
