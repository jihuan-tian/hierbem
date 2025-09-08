// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hmatrix-link-nodes-on-cross-from-diagonal-blocks.cu
 * @brief Link all \hmatrix nodes on a same level and on a same row as well as
 * on a same column with respect to a diagonal block.
 *
 * @ingroup test_cases hierarchical_matrices
 * @author Jihuan Tian
 * @date 2023-11-11
 */

#include <iostream>

#include "hbem_octave_wrapper.h"
#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace HierBEM;
using namespace std;

int
main()
{
  HBEMOctaveWrapper  &inst    = HBEMOctaveWrapper::get_instance();
  auto                oct_val = inst.eval_string("reshape(1:32*32, 32, 32)");
  std::vector<double> values;
  unsigned int        n;
  oct_val.matrix_value(values, n, n);

  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

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
  const unsigned int fixed_rank = 2;
  HMatrix<3, double> H(bct, M, fixed_rank);

  H.link_hmat_nodes_on_cross_from_diagonal_blocks(H.get_property());
  H.print_matrix_info_as_dot(std::cout);

  return 0;
}
