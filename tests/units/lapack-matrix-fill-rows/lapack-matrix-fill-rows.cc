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
 * \file lapack-matrix-fill-rows.cc
 * \brief Verify filling the rows of a @p LAPACKFullMatrixExt.
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2022-11-02
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
    LAPACKFullMatrixExt<double> M(10, 4);
    M.fill_rows({1, 11}, M_src, {5, 8}, 2.0, false);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  return 0;
}
