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
 * \file transpose.cc
 * \brief Test in-place transpose of a LAPACKFullMatrixExt.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-20
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> values{1., 2., 3., 4., 5., 6.};

  LAPACKFullMatrixExt<double> A;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, values, A);

  std::cout << "A=\n";
  A.print_formatted(std::cout, 2, false, 5, "0");

  A.transpose();

  std::cout << "A^T=\n";
  A.print_formatted(std::cout, 2, false, 5, "0");

  return 0;
}
