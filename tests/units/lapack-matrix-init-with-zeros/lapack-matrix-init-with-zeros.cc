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
 * \file lapack-matrix-init-with-zeros.cc
 * \brief Verify if a LAPACKFullMatrixExt is initialized with zeros.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-09-28
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  LAPACKFullMatrixExt<double> A(5, 5);
  A.print_formatted(std::cout, 5, false, 10, "0");

  return 0;
}
