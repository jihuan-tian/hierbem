// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file hmatrix-set-property.cc
 * \brief Verify setting \hmatrix property recursively.
 *
 * \ingroup test_cases hierarchical_matrices
 * \author Jihuan Tian
 * \date 2022-04-19
 */

#include <fstream>
#include <iostream>

#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"
#include "utilities/read_octave_data.h"

int
main()
{
  /**
   * Read a full matrix where only the lower triangular part (including the
   * diagonal) is stored.
   */
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");
  in.close();

  /**
   * Generate index set.
   */
  const unsigned int                   n = 32;
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
   * Create the \hmatrix from the source matrix, where only the lower triangular
   * part is stored.
   */
  const unsigned int fixed_rank = 8;

  /**
   * Set the property of the full matrix as symmetric.
   */
  M.set_property(LAPACKSupport::symmetric);
  HMatrix<3, double> H_symmetric(bct, M, fixed_rank);

  std::ofstream out("H_symmetric_from_M.puml");
  H_symmetric.print_matrix_info_as_dot(out);
  out.close();

  /**
   * Set the property of the full matrix as upper triangular.
   */
  M.set_property(LAPACKSupport::upper_triangular);
  HMatrix<3, double> H_upper_triangular(bct, M, fixed_rank);

  out.open("H_upper_triangular_from_M.puml");
  H_upper_triangular.print_matrix_info_as_dot(out);
  out.close();

  /**
   * Set the property of the full matrix as lower triangular.
   */
  M.set_property(LAPACKSupport::lower_triangular);
  HMatrix<3, double> H_lower_triangular(bct, M, fixed_rank);

  out.open("H_lower_triangular_from_M.puml");
  H_lower_triangular.print_matrix_info_as_dot(out);
  out.close();

  /**
   * Call the @p set_property member function of @p HMatrix.
   */
  M.set_property(LAPACKSupport::general);
  HMatrix<3, double> H(bct, M, fixed_rank);

  H.set_property(HMatrixSupport::symmetric);
  out.open("H_set_property_symmetric.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  H.set_property(HMatrixSupport::upper_triangular);
  out.open("H_set_property_upper_triangular.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  H.set_property(HMatrixSupport::lower_triangular);
  out.open("H_set_property_lower_triangular.puml");
  H.print_matrix_info_as_dot(out);
  out.close();

  return 0;
}
