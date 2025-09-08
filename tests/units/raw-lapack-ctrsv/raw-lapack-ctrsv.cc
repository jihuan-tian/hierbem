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
 * \file raw-lapack-ctrsv.cc
 * \brief Verify the LAPACK function \p ctrsv for solving triangular system.
 * \ingroup test_cases lapack
 * \author Jihuan Tian
 * \date 2021-10-14
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "linear_algebra/lapack_templates_ext.h"
#include "utilities/debug_tools.h"

int
main()
{
  std::vector<std::complex<float>> L_values{{1, 2},
                                            {2, 4},
                                            {4, 8},
                                            {7, 14},
                                            0,
                                            {3, 6},
                                            {5, 10},
                                            {8, 16},
                                            0,
                                            0,
                                            {6, 12},
                                            {9, 18},
                                            0,
                                            0,
                                            0,
                                            {10, 20}};
  std::vector<std::complex<float>> x{{3, 7}, {6, 4}, {9, 7}, {10, 5}};

  char                    uplo{'L'};
  char                    trans{'N'};
  char                    diag{'N'};
  dealii::types::blas_int n    = 4;
  dealii::types::blas_int lda  = 4;
  dealii::types::blas_int incx = 1;

  trsv(&uplo, &trans, &diag, &n, L_values.data(), &lda, x.data(), &incx);

  print_vector_values(std::cout, x, ",", true);

  return 0;
}
