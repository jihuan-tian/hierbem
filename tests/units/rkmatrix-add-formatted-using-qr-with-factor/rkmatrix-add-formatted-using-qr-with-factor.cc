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
 * \file rkmatrix-add-formatted-using-qr-with-factor.cc
 *
 * \brief Verify the formatted addition of two rank-k matrices \fC = A + b B$\f$
 * by using the QR decomposition. This requires that the component matrices of
 * rank-k matrices should be wide matrix, i.e. having more rows than columns.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-05
 */

#include <iostream>
#include <string>

#include "hmatrix/rkmatrix.h"
#include "utilities/debug_tools.h"

int
main()
{
  /**
   * Create two full matrices as the data source.
   */
  LAPACKFullMatrixExt<double> M1, M2;

  std::vector<double> values1{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 1,  2,  3,  4,  5,  6,
                              7,  8,  9,  10, 11, 12, 13, 14, 15, 1,  2,  3,
                              4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 1,  2,  3,  4,  5,  6,
                              7,  8,  9,  10, 11, 12, 13, 14, 15, 1,  2,  3,
                              4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(12, 10, values1, M1);
  M1.print_formatted_to_mat(std::cout, "M1", 8, false, 16, "0");

  std::vector<double> values2{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};
  LAPACKFullMatrixExt<double>::Reshape(12, 10, values2, M2);
  M2.print_formatted_to_mat(std::cout, "M2", 8, false, 16, "0");

  /**
   * Create two rank-k matrices converted from the two matrices.
   */
  const unsigned int rank = 5;

  RkMatrix<double> A(rank, M1);
  A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");

  RkMatrix<double> B(rank, M2);
  B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");

  /**
   * Perform formatted addition.
   */
  double b = 3.5;
  for (unsigned int i = 1; i <= 10; i++)
    {
      RkMatrix<double> C;
      A.add(C, b, B, i);
      C.print_formatted_to_mat(std::cout,
                               std::string("C_trunc_") + std::to_string(i),
                               8,
                               false,
                               16,
                               "0");
    }

  return 0;
}
