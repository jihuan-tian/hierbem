// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file lapack-matrix-keep-first-n-columns.cc
 * @brief
 *
 * @ingroup linalg test_cases
 * @author
 * @date 2024-01-31
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/generic_functors.h"
#include "utilities/unary_template_arg_containers.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  std::vector<double> v(35);
  gen_linear_indices<vector_uta, double>(v, 1, 0.5);
  LAPACKFullMatrixExt<double> A, B;

  for (int i = 5; i >= 0; i--)
    {
      LAPACKFullMatrixExt<double>::Reshape(7, 5, v, A);
      A.keep_first_n_columns(i, false);
      A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");
    }

  for (int i = 5; i >= 0; i--)
    {
      LAPACKFullMatrixExt<double>::Reshape(7, 5, v, B);
      B.keep_first_n_columns(i, true);
      B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");
    }

  return 0;
}
