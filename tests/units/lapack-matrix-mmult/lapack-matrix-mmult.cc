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
 * \file lapack-matrix-mmult.cc
 * \brief Verify the multiplication of two \p LAPACKFullMatrixExt.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-08-19
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> a{1, 3, 5, 7, 9, 10};
  std::vector<double> b{2, 8, 9, 7, 1, 3, 11, 20, 13};
  std::vector<double> c{1, 1, 1, 2, 2, 2};

  LAPACKFullMatrixExt<double> A, B, C, C_adding;

  LAPACKFullMatrixExt<double>::Reshape(2, 3, a, A);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, b, B);
  LAPACKFullMatrixExt<double>::Reshape(2, 3, c, C_adding);

  A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");
  B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");
  C_adding.print_formatted_to_mat(
    std::cout, "C_adding_before", 8, false, 16, "0");

  A.mmult(C, B);
  A.mmult(C_adding, B, true);

  C.print_formatted_to_mat(std::cout, "C", 8, false, 16, "0");
  C_adding.print_formatted_to_mat(std::cout, "C_adding", 8, false, 16, "0");
}
