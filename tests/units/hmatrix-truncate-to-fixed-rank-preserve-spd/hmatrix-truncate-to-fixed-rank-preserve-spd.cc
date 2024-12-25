/**
 * \file hmatrix-truncate-to-fixed-rank-preserve-spd.cc
 * \brief Verify the rank truncation of an \hmatrix, while preserving its
 * \spd_n (SPD) property.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-12-02
 */

#include <fstream>
#include <iostream>

#include "debug_tools.h"
#include "hmatrix/hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

int
main()
{
  // Read a full matrix where only the lower triangular part (including the
  // diagonal) is stored.
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
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

  const unsigned int fixed_rank = 6;
  /**
   * Create the \hmatrix from the full matrix with direct rank truncation.
   */
  HMatrix<3, double> H_no_spd(bct, M, fixed_rank);
  std::ofstream      H_bct("H_bct_no_spd.dat");
  H_no_spd.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  LAPACKFullMatrixExt<double> H_full_no_spd;
  H_no_spd.convertToFullMatrix(H_full_no_spd);
  H_full_no_spd.print_formatted_to_mat(
    std::cout, "H_full_no_spd", 15, false, 25, "0");

  /**
   * Create the \hmatrix from the full matrix without rank truncation.
   */
  HMatrix<3, double> H(bct, M);

  H_bct.open("H_bct_before_trunc.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  std::ofstream H_graph("H_graph_before_trunc.puml");
  H.print_matrix_info_as_dot(H_graph);
  H_graph.close();

  LAPACKFullMatrixExt<double> H_full_before_trunc;
  H.convertToFullMatrix(H_full_before_trunc);
  H_full_before_trunc.print_formatted_to_mat(
    std::cout, "H_full_before_trunc", 15, false, 25, "0");

  /**
   * Next, we perform \spd_n preserving rank truncation.
   */
  H.truncate_to_rank_preserve_positive_definite(fixed_rank);

  H_bct.open("H_bct_after_trunc.dat");
  H.write_leaf_set_by_iteration(H_bct);
  H_bct.close();

  H_graph.open("H_graph_after_trunc.puml");
  H.print_matrix_info_as_dot(H_graph);
  H_graph.close();

  LAPACKFullMatrixExt<double> H_full_after_trunc;
  H.convertToFullMatrix(H_full_after_trunc);
  H_full_after_trunc.print_formatted_to_mat(
    std::cout, "H_full_after_trunc", 15, false, 25, "0");

  return 0;
}
