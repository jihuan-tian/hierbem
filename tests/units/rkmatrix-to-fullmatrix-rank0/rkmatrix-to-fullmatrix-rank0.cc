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
 * \file rkmatrix-to-fullmatrix-rank0.cc
 * \brief Verify the conversion of a rank-0 matrix to a full matrix.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"

int
main()
{
  RkMatrix<double>            M_rk(3, 5, 0);
  LAPACKFullMatrixExt<double> M;
  M_rk.convertToFullMatrix(M);

  M.print_formatted_to_mat(std::cout, "M", 5, false, 8, "0");

  return 0;
}
