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
 * \file rkmatrix-memory-consumption.cc
 * \brief Verify the memory consumption calculation for a rank-k matrix.
 *
 * \ingroup test_cases rkmatrices
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"

using namespace HierBEM;

int
main()
{
  LAPACKFullMatrixExt<double> M;
  std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M);
  std::cout << "M=\n";
  M.print_formatted(std::cout, 5, false, 10, "0");

  std::array<types::global_dof_index, 2> tau{{0, 2}};
  std::array<types::global_dof_index, 2> sigma{{1, 4}};

  RkMatrix<double> A(tau, sigma, 2, M);
  std::cout << "Rank-2 matrix:\n";
  A.print_formatted(std::cout, 5, false, 10, "0");

  std::cout << "Memory consumption of A.A: " << A.get_A().memory_consumption()
            << "\n";
  std::cout << "Memory consumption of A.B: " << A.get_B().memory_consumption()
            << "\n";
  std::cout << "Memory consumption of rank-k matrix A: "
            << A.memory_consumption() << std::endl;

  return 0;
}
