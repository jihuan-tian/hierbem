/**
 * @file hmatrix-tvmult-tril.cc
 * @brief Verify \hmatrix transposed matrix-vector multiplication. The \hmatrix
 * is lower triangular.
 *
 * @ingroup testers hierarchical_matrices
 * @author Jihuan Tian
 * @date 2022-12-01
 */

#include <fstream>
#include <iostream>

#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace std;

int
main()
{
  LAPACKFullMatrixExt<double> M;

  ifstream in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

  /**
   * Set the property of the full matrix as lower_triangular.
   */
  M.set_property(LAPACKSupport::Property::lower_triangular);

  /**
   * Read the vector \f$x\f$.
   */
  Vector<double> x;
  in.open("x.dat");
  read_vector_from_octave(in, "x", x);
  in.close();

  const unsigned int p = 6;
  const unsigned int n = std::pow(2, p);

  AssertDimension(M.m(), n);
  AssertDimension(M.m(), M.n());
  AssertDimension(M.n(), x.size());

  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Generate the \hmatrix from the lower triangular full matrix. Its property
   * will
   * automatically be set to @p HMatrixSupport::Property::lower_triangular.
   */
  const unsigned int n_min        = 2;
  const unsigned int fixed_rank_k = 10;

  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  HMatrix<3, double> H(block_cluster_tree,
                       M,
                       fixed_rank_k,
                       HMatrixSupport::diagonal_block);

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 15, false, 25, "0");

  /**
   * Perform transposed matrix-vector multiplication.
   */
  Vector<double> y(n);
  H.Tvmult(y, x, H.get_property());
  print_vector_to_mat(std::cout, "y1", y, false);

  /**
   * Perform transposed matrix-vector multiplication with factor.
   */
  y = 0.;
  H.Tvmult(y, 0.5, x, H.get_property());
  print_vector_to_mat(std::cout, "y2", y, false);

  return 0;
}
