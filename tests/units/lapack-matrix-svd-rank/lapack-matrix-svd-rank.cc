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
 * \file lapack-matrix-svd-rank.cc
 * \brief Verify matrix rank calculation using SVD.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-04
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  {
    LAPACKFullMatrixExt<double> M;
    std::vector<double> values{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};
    LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M);
    M.print_formatted_to_mat(std::cout, "M1", 4, false, 10, "0");
    std::cout << "rank(M)=" << M.rank() << std::endl;
  }

  {
    LAPACKFullMatrixExt<double> M;
    std::vector<double> values{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};
    LAPACKFullMatrixExt<double>::Reshape(5, 3, values, M);
    M.print_formatted_to_mat(std::cout, "M2", 4, false, 10, "0");
    std::cout << "rank(M)=" << M.rank() << std::endl;
  }

  {
    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    LAPACKFullMatrixExt<double>::Reshape(4, 4, values, M);
    M.print_formatted_to_mat(std::cout, "M3", 4, false, 10, "0");
    std::cout << "rank(M)=" << M.rank() << std::endl;
  }

  {
    LAPACKFullMatrixExt<double> M;
    std::vector<double> values{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    LAPACKFullMatrixExt<double>::Reshape(4, 4, values, M);
    M.print_formatted_to_mat(std::cout, "M4", 4, false, 10, "0");
    std::cout << "rank(M)=" << M.rank() << std::endl;
  }

  return 0;
}
