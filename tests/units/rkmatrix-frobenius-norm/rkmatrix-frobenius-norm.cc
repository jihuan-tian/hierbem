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
 * \file rkmatrix-frobenius-norm.cc
 * \brief Verify the calculation of the Frobenius norm for a rank-k matrix.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-03-15
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/unary_template_arg_containers.h"

int
main()
{
  std::vector<double> a(36);
  gen_linear_indices<vector_uta, double>(a, 1.3, 2.2);
  std::vector<double> b(36);
  gen_linear_indices<vector_uta, double>(b, 20.7, -3.2);

  LAPACKFullMatrixExt<double> A, B;
  LAPACKFullMatrixExt<double>::Reshape(9, 4, a, A);
  LAPACKFullMatrixExt<double>::Reshape(9, 4, b, B);

  A.print_formatted_to_mat(std::cout, "A", 15, false, 25, "0");
  B.print_formatted_to_mat(std::cout, "B", 15, false, 25, "0");

  RkMatrix<double> rkmat(A, B);
  std::cout.precision(15);
  std::cout << "Frobenius norm=" << std::scientific << rkmat.frobenius_norm()
            << std::endl;
  std::cout << "Frobenius norm (2 columns)=" << std::scientific
            << rkmat.frobenius_norm(2) << std::endl;

  return 0;
}
