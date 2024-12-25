/**
 * \file hmatrix-coarsening.cc
 * \brief Coarsen a H-matrix to its subtree.
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-07-19
 */

#include <fstream>
#include <iostream>

#include "block_cluster_tree.h"
#include "debug_tools.h"
#include "hmatrix/hmatrix.h"

int
main()
{
  /**
   * Set the dimension of the H^p-matrix to be built.
   */
  const unsigned int p = 5;
  const unsigned int n = std::pow(2, p);

  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set[i] = i;
    }

  /**
   * Set the minimum cluster size.
   */
  const unsigned int n_min = 2;

  /**
   * Generate the cluster tree using cardinality based partition.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate two block cluster trees. One is coarser than the other.
   */
  BlockClusterTree<3, double> block_cluster_tree_fine(cluster_tree,
                                                      cluster_tree,
                                                      2);
  BlockClusterTree<3, double> block_cluster_tree_coarse(cluster_tree,
                                                        cluster_tree,
                                                        8);

  /**
   * Verify the block cluster partition structures for the two trees by printing
   * out their leaf sets.
   */
  block_cluster_tree_fine.partition_fine_non_tensor_product();
  block_cluster_tree_coarse.partition_fine_non_tensor_product();

  std::ofstream out1("bct1.dat");
  block_cluster_tree_fine.write_leaf_set(out1);
  out1.close();

  std::ofstream out2("bct2.dat");
  block_cluster_tree_coarse.write_leaf_set(out2);
  out2.close();

  /**
   * Create a full matrix as the data source.
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
   * Create an \f$\mathcal{H}\f$-matrix from the full matrix based on the fine
   * block cluster tree.
   */
  const unsigned int fixed_rank = 2;
  HMatrix<3, double> hmat(block_cluster_tree_fine, M, fixed_rank);
  std::ofstream      hmat_fine_out("hmat_fine_partition.dat");
  hmat.write_leaf_set(hmat_fine_out);
  hmat_fine_out.close();

  /**
   * Convert the H-matrix back to full matrix for verification in
   * postprocessing.
   */
  LAPACKFullMatrixExt<double> hmat_to_full;
  hmat.convertToFullMatrix(hmat_to_full);
  hmat_to_full.print_formatted_to_mat(
    std::cout, "hmat_fine_to_full", 8, false, 16, "0");

  /**
   * Coarsen the H-matrix to the subtree.
   */
  const unsigned int fixed_rank_new = 1;
  hmat.coarsen_to_subtree(block_cluster_tree_coarse, fixed_rank_new);
  std::ofstream hmat_coarse_out("hmat_coarse_partition.dat");
  hmat.write_leaf_set(hmat_coarse_out);
  hmat_coarse_out.close();
  hmat.convertToFullMatrix(hmat_to_full);
  hmat_to_full.print_formatted_to_mat(
    std::cout, "hmat_coarse_to_full", 8, false, 16, "0");

  return 0;
}
