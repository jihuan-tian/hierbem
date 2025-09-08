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
 * \file scale-rows-and-columns.cc
 * \brief Test scaling rows and columns of a LAPACKFullMatrixExt, which is
 * actually left and right multiplication with a diagonal matrix.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-20
 */

#include <deal.II/lac/vector.h>

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  /**
   * Diagonal matrix.
   */
  std::vector<double>    V1{1., 2., 3., 4., 5.};
  dealii::Vector<double> V2(V1.begin(), V1.end());

  /**
   * Matrix dimension.
   */
  const unsigned int dim = 5;

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.scale_rows(V1);
    std::cout << "V1*A=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.LAPACKFullMatrix<double>::scale_rows(V2);
    std::cout << "V2*A=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.scale_columns(V1);
    std::cout << "A*V1=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.scale_columns(V2);
    std::cout << "A*V2=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  return 0;
}
