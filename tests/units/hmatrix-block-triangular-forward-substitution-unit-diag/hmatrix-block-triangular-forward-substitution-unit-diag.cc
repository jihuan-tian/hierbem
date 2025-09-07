/**
 * \file hmatrix-block-triangular-forward-substitution-unit-diag.cc
 * \brief Verify forward substitution of a lower unit block triangle \hmatrix.
 * The \bct partition structure is fine non-tensor product.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-22
 */

#include <fstream>
#include <iostream>

#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"
#include "utilities/read_octave_data.h"

int
main()
{
  LAPACKFullMatrixExt<double> L;
  std::ifstream               in1("L.dat");
  L.read_from_mat(in1, "L");
  in1.close();

  Vector<double> b;
  std::ifstream  in2("b.dat");
  read_vector_from_octave(in2, "b", b);
  in2.close();

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
  const unsigned int fixed_rank = 6;
  HMatrix<3, double> H(bct, L, fixed_rank);
  std::ofstream      H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  /**
   * Solve the matrix using forward substitution for unit block triangular
   * matrices.
   */
  Vector<double> x;
  H.solve_block_triangular_by_forward_substitution(x, b, true);

  /**
   * Print the result vector which has overwritten \p b.
   */
  print_vector_to_mat(std::cout, "x", x);
}
