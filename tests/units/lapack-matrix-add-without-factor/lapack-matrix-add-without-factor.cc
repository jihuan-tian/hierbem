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
 * \file lapack-matrix-add-without-factor.cc
 * \brief Verify matrix addition \f$C = A + B\f$
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-05-03
 */

#include <cmath>
#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace HierBEM;

int
main()
{
  std::vector<double> A_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B_data{3, 5, 7, 4, 6, 8, 5, 7, 9};

  LAPACKFullMatrixExt<double> A, B;
  LAPACKFullMatrixExt<double>::Reshape(3, 3, A_data, A);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, B_data, B);

  A.print_formatted_to_mat(std::cout, "A", 8, false, 12, "0");
  B.print_formatted_to_mat(std::cout, "B", 8, false, 12, "0");

  /**
   * Add matrix @p B into @p A.
   */
  A.add(B);
  A.print_formatted_to_mat(std::cout, "A_self_added", 8, false, 12, "0");

  return 0;
}
