/**
 * \file hmatrix-solve-lu.cc
 * \brief Verify LU factorization of an \hmatrix and solve this matrix using
 * forward and backward substitution.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-02
 */

#include <fstream>
#include <iostream>

#include "debug_tools.hcu"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

using namespace HierBEM;

int
main()
{
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
   * Create the \hmatrix.
   */
  const unsigned int fixed_rank = 8;
  HMatrix<3, double> H(bct, M, fixed_rank);

  std::ofstream H_bct("H_bct.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_full", 15, false, 25, "0");

  HMatrix<3, double> LU(bct, fixed_rank);

  /**
   * Perform LU factorization.
   */
  H.compute_lu_factorization(LU, fixed_rank);
  std::cout << "H's state after LU factorization: "
            << HMatrixSupport::state_name(H.get_state()) << "\n";
  std::cout << "LU's state after LU factorization: "
            << HMatrixSupport::state_name(LU.get_state()) << std::endl;

  /**
   * Print the \bct structure of the LU \hmatrix.
   */
  std::ofstream LU_bct("LU_bct.dat");
  LU.write_leaf_set_by_iteration(LU_bct);
  LU_bct.close();

  /**
   * Convert the \Hcal-LU matrix to full matrix.
   */
  LAPACKFullMatrixExt<double> LU_full;
  LU.convertToFullMatrix(LU_full);
  LU_full.print_formatted_to_mat(std::cout, "LU_full", 15, false, 25, "0");

  /**
   * Solve the matrix.
   */
  Vector<double> x;
  LU.solve_lu(x, b);

  /**
   * Print the result vector.
   */
  print_vector_to_mat(std::cout, "x", x);
}
