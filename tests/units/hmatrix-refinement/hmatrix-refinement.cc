// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file hmatrix-refinement.cc
 * \brief Verify the refinement of an \f$\mathcal{H}\f$-matrix hierarchy with
 * respect to its extended block cluster tree.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-07-29
 */

#include <fstream>
#include <iostream>

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
  const unsigned int n_min = 1;

  /**
   * Generate the cluster tree using cardinality based partition.
   */
  ClusterTree<3> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate two block cluster trees with different depth.
   */
  BlockClusterTree<3, double> bct_coarse(cluster_tree, cluster_tree, 4);
  BlockClusterTree<3, double> bct_fine(cluster_tree, cluster_tree, 2);

  bct_coarse.partition_fine_non_tensor_product();
  bct_fine.partition_fine_non_tensor_product();

  /**
   * Print out the partition structures of the two block cluster trees.
   */
  std::ofstream bct_coarse_out("bct-coarse.dat");
  bct_coarse.write_leaf_set(bct_coarse_out);
  std::ofstream bct_fine_out("bct-fine.dat");
  bct_fine.write_leaf_set(bct_fine_out);

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
   * Create an \f$\mathcal{H}\f$-matrix based on the coarse block cluster tree.
   */
  const unsigned int fixed_rank_k = 1;
  HMatrix<3, double> hmat(bct_coarse, M, fixed_rank_k);
  std::ofstream      hmat_coarse_out("hmat-coarse.dat");
  hmat.write_leaf_set_by_iteration(hmat_coarse_out);

  /**
   * Transform the \f$\mathcal{H}\f$-matrix to full matrix for verification.
   */
  LAPACKFullMatrixExt<double> M_from_hmat_coarse;
  hmat.convertToFullMatrix(M_from_hmat_coarse);
  M_from_hmat_coarse.print_formatted_to_mat(
    std::cout, "M_from_hmat_coarse", 8, false, 16, "0");

  /**
   * Extend the block cluster tree associated with the
   * \f$\mathcal{H}\f$-matrix to a finer partition.
   */
  bct_coarse.extend_to_finer_partition(bct_fine.get_leaf_set());

  /**
   * Refine the \f$\mathcal{H}\f$-matrix.
   */
  hmat.refine_to_supertree();
  std::ofstream hmat_fine_out("hmat-fine.dat");
  hmat.write_leaf_set_by_iteration(hmat_fine_out);

  /**
   * Transform the \f$\mathcal{H}\f$-matrix to full matrix for verification.
   */
  LAPACKFullMatrixExt<double> M_from_hmat_fine;
  hmat.convertToFullMatrix(M_from_hmat_fine);
  M_from_hmat_fine.print_formatted_to_mat(
    std::cout, "M_from_hmat_fine", 8, false, 16, "0");

  return 0;
}
