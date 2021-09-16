/**
 * \file hmatrix-invert-by-gauss-elim.cc
 * \brief Verify the inverse of an \hmat using Gauss elimination.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-09-02
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "hmatrix.h"

int
main()
{
  const unsigned int n = 8;

  std::vector<double> M_data{-1, 5,  2,  -3, 6,  1,  -2, 4,  2,  -3, -4, 1,  -3,
                             -1, 1,  2,  -2, 4,  2,  -1, 3,  1,  -1, 3,  -3, 7,
                             2,  -3, 7,  2,  -2, 2,  1,  0,  0,  -1, 1,  -4, 0,
                             0,  0,  2,  0,  -2, 3,  -1, -1, 6,  -2, 4,  3,  -2,
                             4,  -1, -1, 3,  3,  -4, -6, 1,  -3, -3, 1,  -2};
  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(8, 8, M_data, M);
  M.print_formatted_to_mat(std::cout, "M", 8, false, 16, "0");

  /**
   * Generate index set.
   */
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Generate cluster tree.
   */
  const unsigned int n_min = 1;
  ClusterTree<3>     cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree via fine structured non-tensor product
   * partition.
   */
  const unsigned int          n_min_bct = 2;
  BlockClusterTree<3, double> bct(cluster_tree, cluster_tree, n_min_bct);
  bct.partition_coarse_non_tensor_product();

  /**
   * Create the \hmat from the full matrix \p M.
   */
  const unsigned int fixed_rank = 2;
  HMatrix<3, double> H(bct, M, fixed_rank);
  HMatrix<3, double> H_inv;
  InitAndCreateHMatrixChildrenWithoutAlloc(&H_inv, bct.get_root(), true);
  H.invert_by_gauss_elim(H_inv, H, bct, fixed_rank);

  std::ofstream out("H_bct.dat");
  H.write_leaf_set_by_iteration(out);
  out.close();

  /**
   * Convert the inverted matrix into a full matrix for validation.
   */
  LAPACKFullMatrixExt<double> H_full;
  H.convertToFullMatrix(H_full);
  H_full.print_formatted_to_mat(std::cout, "H_inv_full", 12, true, 20, "0");

  return 0;
}
