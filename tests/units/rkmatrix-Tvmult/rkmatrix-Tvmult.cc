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
 * \file rkmatrix-Tvmult.cc
 * \brief Verify the multiplication of a transposed rank-k matrix with a vector.
 * \ingroup rkmatrices
 *
 * \author Jihuan Tian
 * \date 2025-03-25
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
#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify Tvmult for RkMatrix", "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  const unsigned int n = 6;

  LAPACKFullMatrixExt<double> A(n, n);

  unsigned int counter = 1;
  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A(i, j) = (double)counter;
          counter++;
        }
    }

  LAPACKFullMatrixExt<std::complex<double>> A_complex(n, n);

  for (unsigned int i = 0; i < n; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A_complex(i, j) = std::complex<double>(std::sin((double)(i + 1)),
                                                 std::cos((double)(j + 1)));
        }
    }

  RkMatrix<double>               A_rk(2, A);
  RkMatrix<std::complex<double>> A_complex_rk(2, A_complex);

  A_rk.print_formatted(std::cout, 8, false, 15, "0");
  A_complex_rk.print_formatted(std::cout, 8, false, 25, "0");

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
  A_rk.Tvmult(y1, x);
  compare_with_jl_array(y1, "y1", 1e-14, 1e-14);

  Vector<double> y2(y);
  A_rk.Tvmult(y2, x, true);
  compare_with_jl_array(y2, "y2", 1e-14, 1e-14);

  Vector<std::complex<double>> y3(n);
  A_complex_rk.Tvmult(y3, x_complex);
  compare_with_jl_array(y3, "y3", 1e-14, 1e-14);

  Vector<std::complex<double>> y4(y_complex);
  A_complex_rk.Tvmult(y4, x_complex, true);
  compare_with_jl_array(y4, "y4", 1e-14, 1e-14);

  INFO("*** test end");
}
