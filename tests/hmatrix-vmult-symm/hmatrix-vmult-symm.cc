/**
 * \file hmatrix-vmult-symm.cc
 * \brief Verify \hmatrix matrix-vector multiplication. The \hmatrix is
 * symmetric and only its lower triangular part is stored.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-05-14
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
   * Set the property of the full matrix as @p symmetric.
   */
  M.set_property(LAPACKSupport::symmetric);

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
   * Generate the \hmatrix from the symmetric full matrix. Its property will
   * automatically be set to @p HMatrixSupport::Property::symmetric.
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
   * Perform matrix-vector multiplication. \alert{When the \hmatrix is
   * symmetric, its symmetry property should be passed as the third argument
   * to @p vmult.}
   */
  Vector<double> y(n);
  H.vmult(y, x, H.get_property());
  print_vector_to_mat(std::cout, "y1", y, false);

  y = 0.;
  H.vmult(y, 0.5, x, H.get_property());
  print_vector_to_mat(std::cout, "y2", y, false);

  return 0;
}
