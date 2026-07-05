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
 * \file rkmatrix-truncate-to-rank-with-error-matrices.cc
 * \brief Verify the truncation of an RkMatrix to a given rank with the error
 * matrix returned.
 * \ingroup rkmatrices
 *
 * \author Jihuan Tian
 * \date 2025-03-27
 */

#include <catch2/catch_all.hpp>

#include <cmath>
#include <complex>

#include "hbem_cpp_validate.h"
#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE(
  "Verify truncation of an RkMatrix to a given rank with error matrices",
  "[linalg]")
{
  INFO("*** test start");

  const unsigned int n = 6;

  // Both full matrices have rank=2.
  LAPACKFullMatrixExt<double>               A(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex(n, n);

  unsigned int counter = 1;
  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A(i, j) = (double)counter;
          counter++;
        }
    }

  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A_complex(i, j) = std::complex<double>(std::sin((double)(i + 1)),
                                                 std::cos((double)(j + 1)));
        }
    }

  for (unsigned int r = 3; r >= 1; r--)
    {
      LAPACKFullMatrixExt<double>               A_copy(A);
      LAPACKFullMatrixExt<std::complex<double>> A_complex_copy(A_complex);

      LAPACKFullMatrixExt<double> C, D;
      RkMatrix<double>            A_rk(r, A_copy, C, D);

      // Here we check <tt>A_rk.A * A_rk.B^T + C * D^T == A</tt>.
      LAPACKFullMatrixExt<double> CD_transpose;
      C.mTmult(CD_transpose, D);
      LAPACKFullMatrixExt<double> A_reconstructed, A_rk_full;
      A_rk.convertToFullMatrix(A_rk_full);
      A_rk_full.add(A_reconstructed, 1.0, CD_transpose);
      compare_lapack_matrices(A, A_reconstructed, 1e-13, 1e-13);

      LAPACKFullMatrixExt<std::complex<double>> C_complex, D_complex;
      RkMatrix<std::complex<double>>            A_complex_rk(r,
                                                  A_complex_copy,
                                                  C_complex,
                                                  D_complex);
      // Here we check <tt>A_rk.A * A_rk.B^H + C * D^H == A</tt>.
      LAPACKFullMatrixExt<std::complex<double>> CD_complex_transpose;
      C_complex.mHmult(CD_complex_transpose, D_complex);
      LAPACKFullMatrixExt<std::complex<double>> A_complex_reconstructed,
        A_complex_rk_full;
      A_complex_rk.convertToFullMatrix(A_complex_rk_full);
      A_complex_rk_full.add(A_complex_reconstructed, 1.0, CD_complex_transpose);
      compare_lapack_matrices(A_complex, A_complex_reconstructed, 1e-13, 1e-13);
    }
}
