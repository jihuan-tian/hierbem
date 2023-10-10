/**
 * \file hmatrix-cholesky-transpose-forward-substitution-matrix-valued.cc
 * \brief Verify matrix-valued forward substitution of the transpose of a upper
 * triangular \hmatrix.
 *
 * \details The problem to be solved is \f$XU=Z\f$, where \f$U = L^T\f$, both
 * \f$X\f$ and \f$Z\f$ are \hmatrices, which have a same \bct structure.
 * However, this structure can be different from that of \f$U\f$. The \bct
 * partition structure is fine non-tensor product.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-11
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
  LAPACKFullMatrixExt<double> L;
  std::ifstream               in("L.dat");
  L.read_from_mat(in, "L");
  in.close();

  LAPACKFullMatrixExt<double> Z;
  in.open("Z.dat");
  Z.read_from_mat(in, "Z");
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
   * Generate block cluster tree for \p L with the two component cluster trees
   * being the same.
   */
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree, n_min);
  bct.partition_fine_non_tensor_product();

  /**
   * Generate block cluster tree for \p Z with the two component cluster trees
   * being the same.
   */
  BlockClusterTree<3, double> bct_rhs(cluster_tree, cluster_tree, n_min * 2);
  bct_rhs.partition_fine_non_tensor_product();

  /**
   * Create the \hmatrix to be solved by converting from the full matrix \p L.
   */
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, L, fixed_rank);
  std::ofstream      out("HL_bct.dat");
  H.write_leaf_set_by_iteration(out);
  out.close();

  /**
   * Create the RHS \hmatrix \p Z by converting from the full matrix \p Z.
   */
  HMatrix<3, double> HZ(bct_rhs, Z, fixed_rank);
  out.open("HZ_bct.dat");
  HZ.write_leaf_set_by_iteration(out);
  out.close();

  /**
   * Create the empty \hmatrix \p X, memory allocated but with no data.
   */
  HMatrix<3, double> HX(bct_rhs, fixed_rank);

  /**
   * Solve the matrix using the matrix valued Cholesky transposed forward
   * substitution.
   */
  H.solve_cholesky_transpose_by_forward_substitution_matrix_valued(HX,
                                                                   HZ,
                                                                   fixed_rank);
  out.open("HX_bct.dat");
  HX.write_leaf_set_by_iteration(out);
  out.close();

  /**
   * Convert the result \hmatrix \p X to a full matrix.
   */
  LAPACKFullMatrixExt<double> X;
  HX.convertToFullMatrix(X);
  X.print_formatted_to_mat(std::cout, "X", 15, false, 25, "0");
}
