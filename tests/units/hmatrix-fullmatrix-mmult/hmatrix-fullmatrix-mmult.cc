/**
 * \file hmatrix-fullmatrix-mmult.cc
 * \brief Verify the H-matrix/full matrix multiplication.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-08-14
 */

#include <iostream>

#include "hmatrix.h"

int
main()
{
  const unsigned int p = 5;
  const unsigned int n = std::pow(2, p);

  /**
   * Generate the DoF index set.
   */
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Construct the cluster tree.
   */
  const unsigned int n_min = 1;
  ClusterTree<3>     cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Construct the block cluster tree using fine non-tensor product partition.
   */
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree, 4);
  bct.partition_fine_non_tensor_product();

  /**
   * Create a full matrix for initializing the \f$\mathcal{H}\f$-matrix.
   */
  LAPACKFullMatrixExt<double> M1(n, n);
  double                      counter = 1.0;
  for (auto it = M1.begin(); it != M1.end(); it++)
    {
      (*it) = std::sin(counter);
      counter += 1.0;
    }
  M1.print_formatted_to_mat(std::cout, "M1", 8, false, 16, "0");

  /**
   * Create an \f$\mathcal{H}\f$-matrix based on the block cluster tree.
   */
  const unsigned int fixed_rank = 2;
  HMatrix<3, double> M1_h(bct, M1, fixed_rank);

  /**
   * Create the second full matrix.
   */
  LAPACKFullMatrixExt<double> M2(n, 20);
  counter = 1.0;
  for (auto it = M2.begin(); it != M2.end(); it++)
    {
      (*it) = std::cos(counter);
      counter += 1.0;
    }
  M2.print_formatted_to_mat(std::cout, "M2", 8, false, 16, "0");

  /**
   * Perform matrix-matrix multiplication and the result is a full matrix.
   */
  LAPACKFullMatrixExt<double> M_full;
  h_f_mmult(M1_h, M2, M_full);
  M_full.print_formatted_to_mat(std::cout, "M_full", 8, false, 16, "0");

  /**
   * Perform matrix-matrix multiplication and the result is a rank-k matrix.
   */
  RkMatrix<double> M_rk;
  h_f_mmult(M1_h, M2, M_rk);
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 8, false, 16, "0");
}
