/**
 * \file fullmatrix-hmatrix-mmult.cc
 * \brief Verify the full matrix/H-matrix multiplication.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-08-14
 */

#include <iostream>

#include "hmatrix/hmatrix.h"

int
main()
{
  /**
   * Create the first full matrix.
   */
  LAPACKFullMatrixExt<double> M1(32, 32);
  double                      counter = 1.0;
  for (auto it = M1.begin(); it != M1.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }
  M1.print_formatted_to_mat(std::cout, "M1", 8, false, 16, "0");

  /**
   * Create the second full matrix for initializing the second H-matrix.
   */
  LAPACKFullMatrixExt<double> M2(32, 32);
  counter = 1.0;
  for (auto it = M2.begin(); it != M2.end(); it++)
    {
      (*it) = std::cos(counter);
      counter += 1.0;
    }
  M2.print_formatted_to_mat(std::cout, "M2", 8, false, 16, "0");

  /**
   * Generate the DoF index set.
   */
  std::vector<types::global_dof_index> index_set(32);

  for (unsigned int i = 0; i < 32; i++)
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
   * Create an \f$\mathcal{H}\f$-matrix based on the block cluster tree.
   */
  const unsigned int fixed_rank = 2;
  HMatrix<3, double> M2_h(bct, M2, fixed_rank);

  /**
   * Perform matrix-matrix multiplication and the result is a full matrix.
   */
  LAPACKFullMatrixExt<double> M_full;
  f_h_mmult(M1, M2_h, M_full);
  M_full.print_formatted_to_mat(std::cout, "M_full", 8, false, 16, "0");

  /**
   * Perform matrix-matrix multiplication and the result is a rank-k matrix.
   */
  RkMatrix<double> M_rk;
  f_h_mmult(M1, M2_h, M_rk);
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 8, false, 16, "0");
}
