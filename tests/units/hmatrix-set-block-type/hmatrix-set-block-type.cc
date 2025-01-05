/**
 * \file hmatrix-set-block-type.cc
 * \brief Verify setting \hmatrix block type recursively.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-04-20
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
  /**
   * Read a full matrix where only the lower triangular part (including the
   * diagonal) is stored.
   */
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

  /**
   * Create the \hmatrix from the source matrix.
   */
  const unsigned int fixed_rank = 8;

  HMatrix<3, double> H(bct, M, fixed_rank, HMatrixSupport::undefined_block);
  std::ofstream      out("H_undefined_block.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  H.set_block_type(HMatrixSupport::diagonal_block);
  out.open("H_diagonal_block.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  H.set_block_type(HMatrixSupport::upper_triangular_block);
  out.open("H_upper_block.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  H.set_block_type(HMatrixSupport::lower_triangular_block);
  out.open("H_lower_block.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  return 0;
}
