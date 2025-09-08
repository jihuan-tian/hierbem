// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file lapack-matrix-Tvmult.cc
 * \brief Verify transposed matrix-vector multiplication.
 * \ingroup linalg
 *
 * \author Jihuan Tian
 * \date 2022-12-01
 */

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/vector.h>

#include <catch2/catch_all.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "hbem_cpp_validate.h"
#include "hbem_julia_cpp_compare.h"
#include "hbem_julia_wrapper.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify Tvmult for LAPACKFullMatrixExt", "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  const unsigned int n = 6;

  LAPACKFullMatrixExt<double> A(n, n);
  LAPACKFullMatrixExt<double> A_symm(n, n);
  LAPACKFullMatrixExt<double> A_tril(n, n);
  LAPACKFullMatrixExt<double> A_triu(n, n);

  unsigned int counter = 1;
  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A(i, j) = (double)counter;
          counter++;
        }
    }

  A.mTmult(A_symm, A);
  Assert(A_symm.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
  A.lower_triangular(A_tril);
  A.upper_triangular(A_triu);

  LAPACKFullMatrixExt<std::complex<double>> A_complex(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex_symm(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex_hermite_symm(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex_tril(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex_triu(n, n);

  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A_complex(i, j) = std::complex<double>(std::sin((double)(i + 1)),
                                                 std::cos((double)(j + 1)));
        }
    }

  A_complex.mTmult(A_complex_symm, A_complex);
  Assert(A_complex_symm.get_property() == LAPACKSupport::Property::symmetric,
         ExcInternalError());
  A_complex.mHmult(A_complex_hermite_symm, A_complex);
  Assert(A_complex_hermite_symm.get_property() ==
           LAPACKSupport::Property::hermite_symmetric,
         ExcInternalError());
  A_complex.lower_triangular(A_complex_tril);
  A_complex.upper_triangular(A_complex_triu);

  Vector<double> x(n);
  double         step_real = 1.0 / (n - 1);
  double         step_imag = 2.0 / (n - 1);
  for (unsigned int i = 0; i < n; i++)
    x(i) = 1.0 + i * step_real;

  Vector<double> y(x);
  y *= 1.2;

  Vector<std::complex<double>> x_complex(n);
  for (unsigned int i = 0; i < n; i++)
    {
      x_complex(i).real(1.0 + i * step_real);
      x_complex(i).imag(3.0 + i * step_imag);
    }
  Vector<std::complex<double>> y_complex(x_complex);
  y_complex *= std::complex<double>(1.1, 2.3);

  Vector<double> y1(n);
  A.Tvmult(y1, x);
  compare_with_jl_array(y1, "y1", 1e-15, 1e-15);

  Vector<double> y2(y);
  A.Tvmult(y2, x, true);
  compare_with_jl_array(y2, "y2", 1e-15, 1e-15);

  Vector<double> y3(n);
  A_symm.Tvmult(y3, x);
  compare_with_jl_array(y3, "y3", 1e-15, 1e-15);

  Vector<double> y4(y);
  A_symm.Tvmult(y4, x, true);
  compare_with_jl_array(y4, "y4", 1e-15, 1e-15);

  Vector<double> y5(n);
  A_tril.Tvmult(y5, x);
  compare_with_jl_array(y5, "y5", 1e-15, 1e-15);

  Vector<double> y6(y);
  A_tril.Tvmult(y6, x, true);
  compare_with_jl_array(y6, "y6", 1e-15, 1e-15);

  Vector<double> y7(n);
  A_triu.Tvmult(y7, x);
  compare_with_jl_array(y7, "y7", 1e-15, 1e-15);

  Vector<double> y8(y);
  A_triu.Tvmult(y8, x, true);
  compare_with_jl_array(y8, "y8", 1e-15, 1e-15);

  Vector<std::complex<double>> y9(n);
  A_complex.Tvmult(y9, x_complex);
  compare_with_jl_array(y9, "y9", 1e-15, 1e-15);

  Vector<std::complex<double>> y10(y_complex);
  A_complex.Tvmult(y10, x_complex, true);
  compare_with_jl_array(y10, "y10", 1e-15, 1e-15);

  Vector<std::complex<double>> y11(n);
  A_complex_symm.Tvmult(y11, x_complex);
  compare_with_jl_array(y11, "y11", 1e-15, 1e-15);

  Vector<std::complex<double>> y12(y_complex);
  A_complex_symm.Tvmult(y12, x_complex, true);
  compare_with_jl_array(y12, "y12", 1e-15, 1e-15);

  Vector<std::complex<double>> y13(n);
  A_complex_hermite_symm.Tvmult(y13, x_complex);
  compare_with_jl_array(y13, "y13", 1e-15, 1e-15);

  Vector<std::complex<double>> y14(y_complex);
  A_complex_hermite_symm.Tvmult(y14, x_complex, true);
  compare_with_jl_array(y14, "y14", 1e-15, 1e-15);

  Vector<std::complex<double>> y15(n);
  A_complex_tril.Tvmult(y15, x_complex);
  compare_with_jl_array(y15, "y15", 1e-15, 1e-15);

  Vector<std::complex<double>> y16(y_complex);
  A_complex_tril.Tvmult(y16, x_complex, true);
  compare_with_jl_array(y16, "y16", 1e-15, 1e-15);

  Vector<std::complex<double>> y17(n);
  A_complex_triu.Tvmult(y17, x_complex);
  compare_with_jl_array(y17, "y17", 1e-15, 1e-15);

  Vector<std::complex<double>> y18(y_complex);
  A_complex_triu.Tvmult(y18, x_complex, true);
  compare_with_jl_array(y18, "y18", 1e-14, 1e-14);

  INFO("*** test end");
}
