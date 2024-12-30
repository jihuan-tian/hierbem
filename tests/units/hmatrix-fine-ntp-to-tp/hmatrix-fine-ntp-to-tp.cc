/**
 * \file hmatrix-fine-ntp-to-tp.cc
 * \brief Verify the conversion of an \f$\mathcal{H}\f$-matrix from fine
 * non-tensor product structure to tensor product structure.
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-07-30
 */

#include <fstream>
#include <iostream>

#include "hmatrix/hmatrix.h"

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
  BlockClusterTree<3, double> bct_fine_ntp(cluster_tree, cluster_tree, 4);
  bct_fine_ntp.partition_fine_non_tensor_product();
  std::ofstream bct_fine_ntp_out("bct-fine-ntp.dat");
  bct_fine_ntp.write_leaf_set(bct_fine_ntp_out);

  /**
   * Construct the block cluster tree using tensor product partition.
   */
  BlockClusterTree<3, double> bct_tp(cluster_tree, cluster_tree, 8);
  bct_tp.partition_tensor_product();
  std::ofstream bct_tp_out("bct-tp.dat");
  bct_tp.write_leaf_set(bct_tp_out);

  /**
   * Create a full matrix for initializing \f$\mathcal{H}\f$-matrices.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = std::sin(counter);
      counter += 1.0;
    }
  M.print_formatted_to_mat(std::cout, "M", 8, false, 16, "0");

  /**
   * Create an \f$\mathcal{H}\f$-matrix based on the first block cluster tree.
   */
  const unsigned int fixed_rank1 = 2;
  HMatrix<3, double> hmat(bct_fine_ntp, M, fixed_rank1);
  std::ofstream      hmat1_out("hmat1.dat");
  hmat.write_leaf_set_by_iteration(hmat1_out);

  /**
   * Convert \f$\mathcal{H}\f$-matrix back to full matrix for verification.
   */
  LAPACKFullMatrixExt<double> M_from_hmat1;
  hmat.convertToFullMatrix(M_from_hmat1);
  M_from_hmat1.print_formatted_to_mat(
    std::cout, "M_from_hmat1", 8, false, 16, "0");

  /**
   * Convert the \f$\mathcal{H}\f$-matrix to the second block cluster tree.
   */
  const unsigned int fixed_rank2 = 1;
  hmat.convert_between_different_block_cluster_trees(bct_fine_ntp,
                                                     bct_tp,
                                                     fixed_rank2);
  std::ofstream hmat2_out("hmat2.dat");
  hmat.write_leaf_set_by_iteration(hmat2_out);

  /**
   * Convert \f$\mathcal{H}\f$-matrix back to full matrix for verification.
   */
  LAPACKFullMatrixExt<double> M_from_hmat2;
  hmat.convertToFullMatrix(M_from_hmat2);
  M_from_hmat2.print_formatted_to_mat(
    std::cout, "M_from_hmat2", 8, false, 16, "0");

  return 0;
}
