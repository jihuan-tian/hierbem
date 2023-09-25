/**
 * \file hmatrix-fullmatrix-conversion.cc
 * \brief Verify the conversion between H-matrix and full matrix.
 *
 * <dl>
 *   <dt>Comment on results</dt>
 *   <dd>When the truncation rank f$k \geq 4\f$, the approximation error is
 * perfect zero, which contradicts my initial impression, i.e. because the
 * matrix has a dimension of \f$32 \times 32\f$ and is regular (invertible),
 * hence it requires a truncation rank 32 to achieve the perfect
 * approximation. However, after a careful consideration, this initial
 * impression is wrong. Because the original full matrix has been partitioned
 * into smaller blocks and for these blocks, they have much smaller ranks.
 * Therefore, a truncation rank 4 is enough to achieve a perfect
 * approximation.
 *
 * This is verified by a visualization of the block cluster tree structure with
 * matrix block rank displayed.
 * </dd>
 * </dl>
 * \ingroup hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-07-03
 */

#include <fstream>
#include <iostream>
#include <string>

#include "hmatrix.h"

int
main()
{
  const unsigned int p = 5;
  const unsigned int n = std::pow(2, p);

  /**
   * Generate index set.
   */
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min = 2;

  /**
   * Generate cluster tree.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree via fine structured non-tensor product
   * partition.
   */
  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create the full matrix.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter * std::sin(counter);
      counter += 1.0;
    }
  M.print_formatted_to_mat(std::cout, "M", 8, true, 16, "0");

  std::ofstream bct_struct_file("bct-struct-with-rank.dat");
  block_cluster_tree.write_leaf_set(bct_struct_file, M);
  bct_struct_file.close();

  /**
   * Convert the full matrix to H-matrix with specified rank. Then we convert it
   * back to a full matrix for comparison with the original matrix.
   */
  for (unsigned int k = 1; k <= 10; k++)
    {
      HMatrix<3, double>          H(block_cluster_tree, M, k);
      LAPACKFullMatrixExt<double> M_tilde;
      H.convertToFullMatrix(M_tilde);
      M_tilde.print_formatted_to_mat(std::cout,
                                     std::string("M_tilde") + std::to_string(k),
                                     8,
                                     true,
                                     16,
                                     "0");
    }

  return 0;
}
