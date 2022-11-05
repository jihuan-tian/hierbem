/**
 * \file hmatrix-find-diagonal-blocks.cc
 * \brief Find the diagonal blocks related to the current \hmatnode.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-28
 */

#include <fstream>
#include <iostream>

#include "hmatrix.h"

int
main()
{
  std::ofstream out("hmatrix-find-diagonal-blocks.dat");

  const unsigned int                   p = 5;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min        = 2;
  const unsigned int fixed_rank_k = 2;

  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create a full matrix with data.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }

  /**
   * Create the \hmatrix and print its information.
   */
  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);
  hmat.print_matrix_info(out);

  std::ofstream hmat_digraph("hmat.puml");
  hmat.print_matrix_info_as_dot(hmat_digraph);
  hmat_digraph.close();

  HMatrix<3, double> *off_diag_hmat;
  //  /**
  //   * Find the diagonal blocks for the specified \hmatnode when the coarse
  //   * non-tensor product partition is adopted.
  //   */
  //  out << "*** Find the diagonal blocks for the H-matrix node " <<
  //  std::hex
  //            << hmat.get_submatrices()[1] << "\n";
  //  hmat.get_submatrices()[1]
  //    ->find_row_diag_block_for_offdiag_block()
  //    ->print_current_matrix_info(out);
  //  hmat.get_submatrices()[1]
  //    ->find_col_diag_block_for_offdiag_block()
  //    ->print_current_matrix_info(out);
  //
  //  out << "*** Find the diagonal blocks for the H-matrix node " <<
  //  std::hex
  //            << hmat.get_submatrices()[2] << "\n";
  //  hmat.get_submatrices()[2]
  //    ->find_row_diag_block_for_offdiag_block()
  //    ->print_current_matrix_info(out);
  //  hmat.get_submatrices()[2]
  //    ->find_col_diag_block_for_offdiag_block()
  //    ->print_current_matrix_info(out);
  //
  //
  //  out
  //    << "*** Find the diagonal blocks for the H-matrix node " << std::hex
  //    << hmat.get_submatrices()[0]->get_submatrices()[3]->get_submatrices()[2]
  //    << "\n";
  //  hmat.get_submatrices()[0]
  //    ->get_submatrices()[3]
  //    ->get_submatrices()[2]
  //    ->find_row_diag_block_for_offdiag_block()
  //    ->print_current_matrix_info(out);
  //  hmat.get_submatrices()[0]
  //    ->get_submatrices()[3]
  //    ->get_submatrices()[2]
  //    ->find_col_diag_block_for_offdiag_block()
  //    ->print_current_matrix_info(out);

  /**
   * Find the diagonal blocks for the specified \hmatnode when the fine
   * non-tensor product partition is adopted.
   */
  off_diag_hmat =
    hmat.get_submatrices()[0]->get_submatrices()[2]->get_submatrices()[2];
  out << "*** Find the diagonal blocks for the H-matrix node " << std::hex
            << off_diag_hmat << "\n";
  off_diag_hmat->find_row_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);
  off_diag_hmat->find_col_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);

  off_diag_hmat =
    hmat.get_submatrices()[0]->get_submatrices()[2]->get_submatrices()[3];
  out << "*** Find the diagonal blocks for the H-matrix node " << std::hex
            << off_diag_hmat << "\n";
  off_diag_hmat->find_row_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);
  off_diag_hmat->find_col_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);

  off_diag_hmat =
    hmat.get_submatrices()[2]->get_submatrices()[1]->get_submatrices()[0];
  out << "*** Find the diagonal blocks for the H-matrix node " << std::hex
            << off_diag_hmat << "\n";
  off_diag_hmat->find_row_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);
  off_diag_hmat->find_col_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);

  off_diag_hmat =
    hmat.get_submatrices()[2]->get_submatrices()[1]->get_submatrices()[3];
  out << "*** Find the diagonal blocks for the H-matrix node " << std::hex
            << off_diag_hmat << "\n";
  off_diag_hmat->find_row_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);
  off_diag_hmat->find_col_diag_block_for_offdiag_block()
    ->print_current_matrix_info(out);

  out.close();

  return 0;
}
