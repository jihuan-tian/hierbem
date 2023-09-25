/**
 * \file hmatrix-overloaded-shallow-assignment.cc
 * \brief Verify the overloaded shallow assignment operator of
 * \f$\mathcal{H}\f$-matrix.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-08-24
 */

#include <cmath>
#include <fstream>
#include <iostream>

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

  const unsigned int n_min        = 1;
  const unsigned int fixed_rank_k = 2;

  /**
   * Generate cluster tree.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree via fine structured non-tensor product
   * partition.
   */
  BlockClusterTree<3, double> bc_tree(cluster_tree, cluster_tree);
  bc_tree.partition_fine_non_tensor_product();

  /**
   * Create a full matrix as the source data.
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
   * Create an H-matrix from the block cluster tree.
   */
  HMatrix<3, double> H1(bc_tree, M, fixed_rank_k);
  std::cout << "=== H1 ===\n";
  H1.print_formatted(std::cout, 8, false, 16, "0");
  std::ofstream out1("h1.dat");
  H1.write_leaf_set_by_iteration(out1);
  out1.close();

  /**
   * Create an H-matrix using shallow assignment.
   */
  HMatrix<3, double> H2;
  H2 = std::move(H1);
  std::cout << "=== H2 ===\n";
  H2.print_formatted(std::cout, 8, false, 16, "0");
  std::ofstream out2("h2.dat");
  H2.write_leaf_set_by_iteration(out2);
  out2.close();

  /**
   * Try to print H1. Because the data of H1 have been migrated to H2, an
   * exception will be thrown from this function call.
   */
  //! std::cout << "=== H1 after shallow copy ===\n";
  //! H1.print_formatted(std::cout, 8, false, 16, "0");
}
