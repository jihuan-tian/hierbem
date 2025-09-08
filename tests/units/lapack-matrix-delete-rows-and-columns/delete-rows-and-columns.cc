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
 * \file delete-rows-and-columns.cc
 * \brief Test deleting rows and columns as well as keeping the first \p n rows
 * or columns from a LAPACKFullMatrixExt. \ingroup linalg \author Jihuan Tian
 * \date 2021-06-19
 */

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

int
main()
{
  const unsigned int m = 5;
  const unsigned int n = 6;

  LAPACKFullMatrixExt<double> A(m, n);

  unsigned int counter = 1;
  for (unsigned int i = 0; i < m; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A(i, j) = (double)counter;
          counter++;
        }
    }

  std::cout << "A=\n";
  A.print_formatted(std::cout, 2, false, 5, "0");

  {
    LAPACKFullMatrixExt<double> A_del(A);
    A_del.remove_row(3);
    std::cout << "A with row#3 deleted\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double>                            A_del(A);
    std::vector<std::make_unsigned<types::blas_int>::type> row_indices{1, 3, 4};
    A_del.remove_rows(row_indices);
    std::cout << "A with row#1,3,4 deleted\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A_del(A);
    A_del.remove_column(3);
    std::cout << "A with column#3 deleted\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double>                            A_del(A);
    std::vector<std::make_unsigned<types::blas_int>::type> column_indices{1,
                                                                          3,
                                                                          4};
    A_del.remove_columns(column_indices);
    std::cout << "A with column#1,3,4 deleted\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A_del(A);
    A_del.keep_first_n_rows(3, true);
    std::cout << "A with first 3 rows kept while deleting others\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A_del(A);
    A_del.keep_first_n_rows(3, false);
    std::cout << "A with first 3 rows kept while setting others to zero\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A_del(A);
    A_del.keep_first_n_columns(4, true);
    std::cout << "A with first 4 columns kept while deleting others\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A_del(A);
    A_del.keep_first_n_columns(4, false);
    std::cout << "A with first 4 columns kept while setting others to zero\n";
    A_del.print_formatted(std::cout, 2, false, 5, "0");
  }
}
