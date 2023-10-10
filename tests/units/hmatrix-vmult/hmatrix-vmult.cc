/**
 * \file hmatrix-vmult.cc
 * \brief Verify \hmatrix matrix-vector multiplication.
 *
 * \ingroup testers hierarchical_matrices
 * \author Jihuan Tian
 * \date 2021-06-23
 */

#include "debug_tools.h"
#include "hmatrix.h"

int
main()
{
  const unsigned int                   p = 5;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  const unsigned int n_min        = 1;
  const unsigned int fixed_rank_k = 1;

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
  M.print_formatted_to_mat(std::cout, "M");

  /**
   * Create a rank-1 HMatrix.
   */
  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);

  /**
   * Create the vector x.
   */
  Vector<double> x(n);
  for (unsigned int i = 0; i < n; i++)
    {
      x(i) = std::sin(static_cast<double>(n));
    }
  print_vector_to_mat(std::cout, "x", x);

  /**
   * Perform matrix-vector multiplication.
   */
  Vector<double> y(n);
  hmat.vmult(y, x);
  print_vector_to_mat(std::cout, "y1", y);

  y = 0.;
  hmat.vmult(y, 0.5, x);
  print_vector_to_mat(std::cout, "y2", y);
}
