// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <catch2/catch_all.hpp>

#include <complex>
#include <fstream>

#include "hmatrix/hmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/read_octave_data.h"

using namespace HierBEM;
using namespace Catch::Matchers;

void
run_hmatrix_vmult_serial_iterative()
{
  std::ofstream ofs("hmatrix-vmult-serial-iterative.output");

  /**
   * Load a general matrix.
   */
  LAPACKFullMatrixExt<double>               M;
  LAPACKFullMatrixExt<std::complex<double>> M_complex;

  std::ifstream in("M.dat");
  M.read_from_mat(in, "M");
  M_complex.read_from_mat(in, "M_complex");
  in.close();

  REQUIRE(M.size()[0] > 0);
  REQUIRE(M.size()[0] == M.size()[1]);
  REQUIRE(M_complex.size()[0] > 0);
  REQUIRE(M_complex.size()[0] == M_complex.size()[1]);

  /**
   * Generate index set.
   */
  const unsigned int                   p = 6;
  const unsigned int                   n = std::pow(2, p);
  std::vector<types::global_dof_index> index_set(n);

  for (unsigned int i = 0; i < n; i++)
    {
      index_set.at(i) = i;
    }

  /**
   * Generate cluster tree.
   */
  const unsigned int            spacedim = 3;
  const unsigned int            n_min    = 2;
  ClusterTree<spacedim, double> cluster_tree(index_set, n_min);
  cluster_tree.partition();

  /**
   * Generate block cluster tree with the two component cluster trees being the
   * same.
   */
  BlockClusterTree<spacedim, double> block_cluster_tree(cluster_tree,
                                                        cluster_tree);
  block_cluster_tree.partition_fine_non_tensor_product();

  /**
   * Create a rank-k HMatrix.
   */
  const unsigned int        fixed_rank_k = n / 4;
  HMatrix<spacedim, double> H(block_cluster_tree, M, fixed_rank_k);
  HMatrix<spacedim, std::complex<double>> H_complex(block_cluster_tree,
                                                    M_complex,
                                                    fixed_rank_k);

  REQUIRE(H.get_property() == HMatrixSupport::Property::general);
  REQUIRE(H_complex.get_property() == HMatrixSupport::Property::general);
  REQUIRE(H.get_m() == M.size()[0]);
  REQUIRE(H.get_n() == M.size()[1]);
  REQUIRE(H_complex.get_m() == M_complex.size()[0]);
  REQUIRE(H_complex.get_n() == M_complex.size()[1]);

  /**
   * Convert the \hmatrix back to full matrix for comparison with the original
   * full matrix.
   */
  LAPACKFullMatrixExt<double>               H_full;
  LAPACKFullMatrixExt<std::complex<double>> H_full_complex;

  H.convertToFullMatrix(H_full);
  H_complex.convertToFullMatrix(H_full_complex);

  REQUIRE(H_full.size()[0] == M.size()[0]);
  REQUIRE(H_full.size()[1] == M.size()[1]);
  REQUIRE(H_full_complex.size()[0] == M_complex.size()[0]);
  REQUIRE(H_full_complex.size()[1] == M_complex.size()[1]);

  H_full.print_formatted_to_mat(ofs, "H_full", 15, false, 25, "0");
  H_full_complex.print_formatted_to_mat(
    ofs, "H_full_complex", 15, false, 45, "0");

  /**
   * Read the vector \f$x\f$ and the initial values of the vector \f$y\f$.
   */
  Vector<double>               x;
  Vector<std::complex<double>> x_complex;
  Vector<double>               y;
  Vector<std::complex<double>> y_complex;

  in.open("xy.dat");
  read_vector_from_octave(in, "x", x);
  read_vector_from_octave(in, "x_complex", x_complex);
  read_vector_from_octave(in, "y0", y);
  read_vector_from_octave(in, "y0_complex", y_complex);
  in.close();

  REQUIRE(x.size() == M.size()[1]);
  REQUIRE(x_complex.size() == M_complex.size()[1]);
  REQUIRE(y.size() == M.size()[0]);
  REQUIRE(y_complex.size() == M_complex.size()[0]);

  /**
   * Perform \hmatrix/vector multiplication.
   */
  H.vmult_serial_iterative(0.3, y, 1.5, x);
  print_vector_to_mat(ofs, "y1_cpp", y);

  H_complex.vmult_serial_iterative(0.3, y_complex, 1.5, x_complex);
  print_vector_to_mat(ofs, "y1_cpp_complex", y_complex);
  H_complex.vmult_serial_iterative(std::complex<double>(0.3, 0.2),
                                   y_complex,
                                   std::complex<double>(1.5, 2.1),
                                   x_complex);
  print_vector_to_mat(ofs, "y2_cpp_complex", y_complex);
  H_complex.vmult_serial_iterative(std::complex<double>(0.3, 0.2),
                                   y_complex,
                                   1.5,
                                   x_complex);
  print_vector_to_mat(ofs, "y3_cpp_complex", y_complex);
  H_complex.vmult_serial_iterative(0.3,
                                   y_complex,
                                   std::complex<double>(1.5, 2.1),
                                   x_complex);
  print_vector_to_mat(ofs, "y4_cpp_complex", y_complex);

  ofs.close();
}
