/**
 * \file hmatrix-check-parent-and-submatrix-index.cc
 * \brief Check the @p parent and @p submatrix_index member variables of
 * \hmatrix by visualizing the hierarchical structure of the block cluster tree
 * with the help of the tool @p dot in @p Graphviz.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-11-27
 */

#include <fstream>
#include <iostream>

#include "hmatrix/hmatrix.h"

int
main()
{
  std::ofstream out("hmatrix-check-parent-and-submatrix-index.dat");
  
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

  out.close();

  return 0;
}
