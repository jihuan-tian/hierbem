// Copyright (C) 2021-2024 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file hmatrix-lower-write-leaf-set-by-iteration.cu
 * \brief Verify the method for writing out leaf set by iteration over the
 * constructed leaf set instead of recursion. By default, the traversal follows
 * the Z curve.
 *
 * The \hmatrix in this test case is lower triangular.
 *
 * \ingroup
 * \author Jihuan Tian
 * \date 2024-03-11
 */

#include <fstream>
#include <iostream>

#include "hbem_octave_wrapper.h"
#include "hmatrix/hmatrix.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  /**
   * Create the global index set.
   */
  const unsigned int                   p = 4;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Construct the cluster tree.
   */
  const unsigned int n_min = 1;
  ClusterTree<3>     cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Construct the block cluster tree.
   */
  BlockClusterTree<3, double> block_cluster_tree(cluster_tree, cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create a full matrix with data.
   */
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();
  inst.add_path(SOURCE_DIR);
  // Execute script `gen_lower_matrix.m` to generate M.dat
  inst.source_file(SOURCE_DIR "/gen_lower_matrix.m");

  LAPACKFullMatrixExt<double> M;
  ifstream                    in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

  /**
   * Set the property of the full matrix as @p lower_triangular.
   */
  M.set_property(LAPACKSupport::lower_triangular);

  /**
   * Create a rank-1 HMatrix, whose property is automatically set to @p lower_triangular
   */
  const unsigned int fixed_rank_k = 1;
  HMatrix<3, double> hmat(block_cluster_tree, M, fixed_rank_k);

  /**
   * Write out the leaf nodes by iteration.
   */
  hmat.write_leaf_set_by_iteration(std::cout);

  return 0;
}
