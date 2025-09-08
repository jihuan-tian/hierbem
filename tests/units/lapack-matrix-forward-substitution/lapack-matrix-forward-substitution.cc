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
 * \file lapack-matrix-forward-substitution.cc
 * \brief Verify forward substitution of a unit lower triangle matrix.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-16
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> L_values{1, 2, 4, 7, 0, 1, 5, 8, 0, 0, 1, 9, 0, 0, 0, 1};
  LAPACKFullMatrixExt<double> L;
  LAPACKFullMatrixExt<double>::Reshape(4, 4, L_values, L);
  L.set_property(LAPACKSupport::lower_triangular);

  {
    Vector<double> b({3, 6, 9, 10});
    L.solve_by_forward_substitution(b);
    b.print(std::cout, 8);
  }

  {
    Vector<double> b({3, 6, 9, 10});
    Vector<double> x;
    L.solve_by_forward_substitution(x, b);
    x.print(std::cout, 8);
  }
}
