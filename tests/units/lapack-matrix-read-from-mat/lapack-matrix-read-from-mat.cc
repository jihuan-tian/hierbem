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
 * \file lapack-matrix-read-from-mat.cc
 * \brief Verify reading a matrix from a file saved from Octave in text format,
 * i.e. saved with the option \p -text.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-20
 */

#include <fstream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::ifstream input("input.mat");

  LAPACKFullMatrixExt<double> M;
  M.read_from_mat(input, "M");
  M.print_formatted_to_mat(std::cout, "M_read", 15, false, 25, "0");

  return 0;
}
