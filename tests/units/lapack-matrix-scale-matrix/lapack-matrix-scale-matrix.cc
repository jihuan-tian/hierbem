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
 * @file lapack-matrix-scale-matrix.cc
 * @brief Verify LAPACKFullMatrixExt scaled by a factor.
 * @ingroup linalg
 *
 * @date 2025-03-28
 * @author Jihuan Tian
 */

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/lapack_support.h>

#include <catch2/catch_all.hpp>

#include <cmath>
#include <complex>

#include "hbem_julia_cpp_compare.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify LAPACKFullMatrixExt scaled by a factor", "[linalg]")
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
  Assert(A_tril.get_property() == LAPACKSupport::Property::lower_triangular,
         ExcInternalError());
  A.upper_triangular(A_triu);
  Assert(A_triu.get_property() == LAPACKSupport::Property::upper_triangular,
         ExcInternalError());

  LAPACKFullMatrixExt<std::complex<double>> A_complex(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex_symm(n, n);
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
  A_complex.lower_triangular(A_complex_tril);
  Assert(A_complex_tril.get_property() ==
           LAPACKSupport::Property::lower_triangular,
         ExcInternalError());
  A_complex.upper_triangular(A_complex_triu);
  Assert(A_complex_triu.get_property() ==
           LAPACKSupport::Property::upper_triangular,
         ExcInternalError());

  // Scaled by a real factor.
  double factor = 0.3;
  {
    LAPACKFullMatrixExt<double> A_copy(A);
    LAPACKFullMatrixExt<double> A_symm_copy(A_symm);
    Assert(A_symm_copy.get_property() == LAPACKSupport::Property::symmetric,
           ExcInternalError());
    LAPACKFullMatrixExt<double> A_tril_copy(A_tril);
    Assert(A_tril_copy.get_property() ==
             LAPACKSupport::Property::lower_triangular,
           ExcInternalError());
    LAPACKFullMatrixExt<double> A_triu_copy(A_triu);
    Assert(A_triu_copy.get_property() ==
             LAPACKSupport::Property::upper_triangular,
           ExcInternalError());

    A_copy *= factor;
    A_symm_copy *= factor;
    A_tril_copy *= factor;
    A_triu_copy *= factor;

    compare_with_jl_matrix(A_copy, "A*factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_symm_copy, "A_symm*factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_tril_copy, "A_tril*factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_triu_copy, "A_triu*factor", 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> A_complex_copy(A_complex);
    LAPACKFullMatrixExt<std::complex<double>> A_complex_symm_copy(
      A_complex_symm);
    Assert(A_complex_symm_copy.get_property() ==
             LAPACKSupport::Property::symmetric,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_tril_copy(
      A_complex_tril);
    Assert(A_complex_tril_copy.get_property() ==
             LAPACKSupport::Property::lower_triangular,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_triu_copy(
      A_complex_triu);
    Assert(A_complex_triu_copy.get_property() ==
             LAPACKSupport::Property::upper_triangular,
           ExcInternalError());

    A_complex_copy *= factor;
    A_complex_symm_copy *= factor;
    A_complex_tril_copy *= factor;
    A_complex_triu_copy *= factor;

    compare_with_jl_matrix(A_complex_copy, "A_complex*factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_complex_symm_copy,
                           "A_complex_symm*factor",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_tril_copy,
                           "A_complex_tril*factor",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_triu_copy,
                           "A_complex_triu*factor",
                           1e-15,
                           1e-15);
  }

  {
    LAPACKFullMatrixExt<double> A_copy(A);
    LAPACKFullMatrixExt<double> A_symm_copy(A_symm);
    Assert(A_symm_copy.get_property() == LAPACKSupport::Property::symmetric,
           ExcInternalError());
    LAPACKFullMatrixExt<double> A_tril_copy(A_tril);
    Assert(A_tril_copy.get_property() ==
             LAPACKSupport::Property::lower_triangular,
           ExcInternalError());
    LAPACKFullMatrixExt<double> A_triu_copy(A_triu);
    Assert(A_triu_copy.get_property() ==
             LAPACKSupport::Property::upper_triangular,
           ExcInternalError());

    A_copy /= factor;
    A_symm_copy /= factor;
    A_tril_copy /= factor;
    A_triu_copy /= factor;

    compare_with_jl_matrix(A_copy, "A/factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_symm_copy, "A_symm/factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_tril_copy, "A_tril/factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_triu_copy, "A_triu/factor", 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> A_complex_copy(A_complex);
    LAPACKFullMatrixExt<std::complex<double>> A_complex_symm_copy(
      A_complex_symm);
    Assert(A_complex_symm_copy.get_property() ==
             LAPACKSupport::Property::symmetric,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_tril_copy(
      A_complex_tril);
    Assert(A_complex_tril_copy.get_property() ==
             LAPACKSupport::Property::lower_triangular,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_triu_copy(
      A_complex_triu);
    Assert(A_complex_triu_copy.get_property() ==
             LAPACKSupport::Property::upper_triangular,
           ExcInternalError());

    A_complex_copy /= factor;
    A_complex_symm_copy /= factor;
    A_complex_tril_copy /= factor;
    A_complex_triu_copy /= factor;

    compare_with_jl_matrix(A_complex_copy, "A_complex/factor", 1e-15, 1e-15);
    compare_with_jl_matrix(A_complex_symm_copy,
                           "A_complex_symm/factor",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_tril_copy,
                           "A_complex_tril/factor",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_triu_copy,
                           "A_complex_triu/factor",
                           1e-15,
                           1e-15);
  }

  // Scaled by a complex factor.
  std::complex<double> factor_complex(0.3, 0.7);
  {
    LAPACKFullMatrixExt<std::complex<double>> A_complex_copy(A_complex);
    LAPACKFullMatrixExt<std::complex<double>> A_complex_symm_copy(
      A_complex_symm);
    Assert(A_complex_symm_copy.get_property() ==
             LAPACKSupport::Property::symmetric,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_tril_copy(
      A_complex_tril);
    Assert(A_complex_tril_copy.get_property() ==
             LAPACKSupport::Property::lower_triangular,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_triu_copy(
      A_complex_triu);
    Assert(A_complex_triu_copy.get_property() ==
             LAPACKSupport::Property::upper_triangular,
           ExcInternalError());

    A_complex_copy *= factor_complex;
    A_complex_symm_copy *= factor_complex;
    A_complex_tril_copy *= factor_complex;
    A_complex_triu_copy *= factor_complex;

    compare_with_jl_matrix(A_complex_copy,
                           "A_complex*factor_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_symm_copy,
                           "A_complex_symm*factor_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_tril_copy,
                           "A_complex_tril*factor_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_triu_copy,
                           "A_complex_triu*factor_complex",
                           1e-15,
                           1e-15);
  }

  {
    LAPACKFullMatrixExt<std::complex<double>> A_complex_copy(A_complex);
    LAPACKFullMatrixExt<std::complex<double>> A_complex_symm_copy(
      A_complex_symm);
    Assert(A_complex_symm_copy.get_property() ==
             LAPACKSupport::Property::symmetric,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_tril_copy(
      A_complex_tril);
    Assert(A_complex_tril_copy.get_property() ==
             LAPACKSupport::Property::lower_triangular,
           ExcInternalError());
    LAPACKFullMatrixExt<std::complex<double>> A_complex_triu_copy(
      A_complex_triu);
    Assert(A_complex_triu_copy.get_property() ==
             LAPACKSupport::Property::upper_triangular,
           ExcInternalError());

    A_complex_copy /= factor_complex;
    A_complex_symm_copy /= factor_complex;
    A_complex_tril_copy /= factor_complex;
    A_complex_triu_copy /= factor_complex;

    compare_with_jl_matrix(A_complex_copy,
                           "A_complex/factor_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_symm_copy,
                           "A_complex_symm/factor_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_tril_copy,
                           "A_complex_tril/factor_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(A_complex_triu_copy,
                           "A_complex_triu/factor_complex",
                           1e-15,
                           1e-15);
  }

  INFO("*** test end");
}
