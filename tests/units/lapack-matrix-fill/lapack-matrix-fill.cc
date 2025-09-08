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
 * \file lapack-matrix-fill.cc
 * \brief Verify filling a LAPACKFullMatrixExt from a source matrix.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-30
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double>         src_data{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  LAPACKFullMatrixExt<double> M_src;
  LAPACKFullMatrixExt<double>::Reshape(3, 4, src_data, M_src);

  {
    LAPACKFullMatrixExt<double> M(3, 4);
    M.fill(M_src, 1, 2, 1, 1, 1., false);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(3, 4);
    M.fill(M_src, 1, 2, 1, 1, 2., true);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(2, 2);
    M.fill(M_src, 0, 0, 2, 2, 1., false);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(2, 2);
    M.fill(M_src, 0, 0, 2, 2, 1., true);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  return 0;
}
