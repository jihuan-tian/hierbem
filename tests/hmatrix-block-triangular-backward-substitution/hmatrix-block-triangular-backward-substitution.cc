/**
 * \file hmatrix-block-triangular-backward-substitution.cc
 * \brief Verify backward substitution of an upper block triangle \hmatrix. The
 * \bct partition structure is fine non-tensor product.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-10-22
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
  LAPACKFullMatrixExt<double> U;
  std::ifstream               in1("U.dat");
  U.read_from_mat(in1, "U");
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
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, U, fixed_rank);
  std::ofstream      H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  /**
   * Solve the matrix using backward substitution.
   */
  Vector<double> x;
  H.solve_block_triangular_by_backward_substitution(x, b, false);

  /**
   * Print the result vector which has overwritten \p b.
   */
  print_vector_to_mat(std::cout, "x", x);
}
