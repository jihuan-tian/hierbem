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
 * \file lapack-matrix-memory-consumption.cc
 * \brief Verify the memory consumption calculation for a @p LAPACKFullMatrixExt.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/unary_template_arg_containers.h"

using namespace HierBEM;

int
main()
{
  std::cout << "# Matrix dimension,Memory consumption\n";

  {
    unsigned int n = 10;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }

  {
    unsigned int n = 100;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }

  {
    unsigned int n = 1000;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }


  {
    unsigned int n = 10000;

    LAPACKFullMatrixExt<double> M;
    std::vector<double>         values(n * n);
    gen_linear_indices<vector_uta, double>(values, 1, 1.5);
    LAPACKFullMatrixExt<double>::Reshape(n, n, values, M);

    std::cout << n << "," << M.memory_consumption() << std::endl;
  }

  return 0;
}
