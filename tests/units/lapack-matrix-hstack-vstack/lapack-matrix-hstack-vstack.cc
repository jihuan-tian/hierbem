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
 * \file hstack-vstack.cc
 * \brief Verify horizontal and vertical stacking of two LAPACKFullMatrixExt
 * objects.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-30
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  LAPACKFullMatrixExt<double> A, B, C;
  std::vector<double>         A_data{1, 2, 3, 4, 5, 6};
  std::vector<double>         B_data{7, 8, 9, 10, 11, 12};

  LAPACKFullMatrixExt<double>::Reshape(2, 3, A_data, A);
  LAPACKFullMatrixExt<double>::Reshape(2, 3, B_data, B);

  std::cout << "A=\n";
  A.print_formatted(std::cout);

  std::cout << "B=\n";
  B.print_formatted(std::cout);

  std::cout << "Horizontal stacking [A, B]=: \n";
  A.hstack(C, B);
  C.print_formatted(std::cout);

  std::cout << "Vertical stacking [A; B]=: \n";
  A.vstack(C, B);
  C.print_formatted(std::cout);
}
