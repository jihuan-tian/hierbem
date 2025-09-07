/**
 * \file hmatrix-rkmatrix-conversion.cc
 * \brief Verify the conversion from an H-matrix to a rank-k matrix.
 * \ingroup hierarchical_matrices test_cases
 * \author Jihuan Tian
 * \date 2021-07-09
 */

#include <fstream>
#include <iostream>

#include "hmatrix/hmatrix.h"
#include "hmatrix/rkmatrix.h"

int
main()
{
  const unsigned int p = 7;
  const unsigned int n = std::pow(2, p);

  /**
   * Create a full matrix as the data source.
   */
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }
  M.print_formatted_to_mat(std::cout, "M", 8, false, 16, "0");

  /**
   * Create an HMatrix of fixed rank 2 using the fine non-tensor product
   * partition. In practice, the rank may not be a constant but a block
   * dependent distribution or map.
   */

  std::vector<types::global_dof_index> index_set(n);
  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min        = 2;
  const unsigned int fixed_rank_k = 3;

  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);
  // std::cout << "HMatrix rank-3:\n";
  // hmat.print_formatted(std::cout, 8, false, 16, "0");

  /**
   * Print the initial partition structure.
   */
  std::ofstream output_stream("hmat-bct0.dat");
  hmat.write_leaf_set(output_stream);
  output_stream.close();

  /**
   * Convert the H-matrix to a rank-2 matrix.
   */
  size_t calling_counter = 1;
  convertHMatBlockToRkMatrix(&hmat, 2, &hmat, &calling_counter);

  /**
   * Extract the resulted matrix, which is either a rank-k matrix or a full
   * matrix.
   */
  if (hmat.get_type() == FullMatrixType)
    {
      hmat.get_fullmatrix()->print_formatted_to_mat(
        std::cout, "M_agglomerated", 8, false, 16, "0");
    }
  else if (hmat.get_type() == RkMatrixType)
    {
      hmat.get_rkmatrix()->print_formatted_to_mat(
        std::cout, "M_agglomerated", 8, false, 16, "0");
    }
  else
    {
      Assert(false, ExcInvalidHMatrixType(hmat.get_type()));
    }

  return 0;
}
