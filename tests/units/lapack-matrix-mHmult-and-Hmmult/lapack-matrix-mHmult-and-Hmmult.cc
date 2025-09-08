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
 * @file lapack-matrix-mHmult-and-Hmmult.cc
 * @brief Verify mHmult and Hmmult for complex valued LAPACKFullMatrixExt
 * @ingroup linalg
 *
 * @date 2025-03-11
 * @author Jihuan Tian
 */

#include <deal.II/lac/lapack_support.h>

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

TEST_CASE("Verify mHmult and Hmmult for complex valued LAPACKFullMatrixExt",
          "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  const unsigned int m = 3;
  const unsigned int n = 5;

  std::complex<double>                     *mat_ptr;
  LAPACKFullMatrixExt<std::complex<double>> A(m, n);
  mat_ptr = &(A.begin()->value());
  for (unsigned int i = 1; i <= m * n; i++, mat_ptr++)
    {
      (*mat_ptr) =
        std::complex<double>(std::sin((double)i), std::cos((double)i));
    }

  LAPACKFullMatrixExt<std::complex<double>> B(m, n);
  mat_ptr = &(B.begin()->value());
  for (unsigned int i = 1; i <= m * n; i++, mat_ptr++)
    {
      (*mat_ptr) =
        std::complex<double>(std::cos((double)i), std::sin((double)i));
    }

  LAPACKFullMatrixExt<std::complex<double>> A_mul_AH(m, m);
  LAPACKFullMatrixExt<std::complex<double>> AH_mul_A(n, n);
  LAPACKFullMatrixExt<std::complex<double>> A_mul_BH(m, m);
  LAPACKFullMatrixExt<std::complex<double>> AH_mul_B(n, n);

  // Multiplication without adding into the result matrix.
  A.mHmult(A_mul_AH, A);
  REQUIRE(A_mul_AH.get_property() ==
          LAPACKSupport::Property::hermite_symmetric);
  A.Hmmult(AH_mul_A, A);
  REQUIRE(AH_mul_A.get_property() ==
          LAPACKSupport::Property::hermite_symmetric);
  A.mHmult(A_mul_BH, B);
  REQUIRE(A_mul_BH.get_property() == LAPACKSupport::Property::general);
  A.Hmmult(AH_mul_B, B);
  REQUIRE(AH_mul_B.get_property() == LAPACKSupport::Property::general);

  compare_with_jl_matrix(A_mul_AH, "A_mul_AH", 1e-15, 1e-15);
  compare_with_jl_matrix(AH_mul_A, "AH_mul_A", 1e-15, 1e-15);
  compare_with_jl_matrix(A_mul_BH, "A_mul_BH", 1e-15, 1e-15);
  compare_with_jl_matrix(AH_mul_B, "AH_mul_B", 1e-15, 1e-15);

  // Multiplication with adding into the result matrix: alpha=2.5+1.2im
  {
    // Generate a Hermite symmetric matrix @p C1_add_A_mul_AH , but set it as a
    // general matrix.
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AH_tmp(m, m);
    for (unsigned int i = 1; i <= m; i++)
      for (unsigned int j = 1; j <= m; j++)
        C1_add_A_mul_AH_tmp(i - 1, j - 1) =
          std::complex<double>(i / 10., j / 10.);
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AH(m, m);
    C1_add_A_mul_AH_tmp.mHmult(C1_add_A_mul_AH, C1_add_A_mul_AH_tmp);
    C1_add_A_mul_AH.set_property(LAPACKSupport::Property::general);

    // Generate a Hermite symmetric matrix @p C1_add_A_mul_AH_hsymm and set its
    // property as it is.
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AH_hsymm(
      C1_add_A_mul_AH);
    C1_add_A_mul_AH_hsymm.set_property(
      LAPACKSupport::Property::hermite_symmetric);

    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_BH(C1_add_A_mul_AH);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_A_tmp(n, n);
    for (unsigned int i = 1; i <= n; i++)
      for (unsigned int j = 1; j <= n; j++)
        C2_add_AH_mul_A_tmp(i - 1, j - 1) =
          std::complex<double>(i / 10., j / 10.);
    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_A(n, n);
    C2_add_AH_mul_A_tmp.mHmult(C2_add_AH_mul_A, C2_add_AH_mul_A_tmp);
    C2_add_AH_mul_A.set_property(LAPACKSupport::Property::general);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_A_hsymm(
      C2_add_AH_mul_A);
    C2_add_AH_mul_A_hsymm.set_property(
      LAPACKSupport::Property::hermite_symmetric);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_B(C2_add_AH_mul_A);

    std::complex<double> alpha(2.5, 1.2);

    A.mHmult(C1_add_A_mul_AH, alpha, A, true);
    REQUIRE(C1_add_A_mul_AH.get_property() == LAPACKSupport::Property::general);
    // Because @p alpha is complex, the result is not a Hermite symmetric matrix,
    // even though @p C1_add_A_mul_AH_hsymm is an Hermite symmetric matrix to be
    // added into.
    A.mHmult(C1_add_A_mul_AH_hsymm, alpha, A, true);
    REQUIRE(C1_add_A_mul_AH_hsymm.get_property() ==
            LAPACKSupport::Property::general);
    A.Hmmult(C2_add_AH_mul_A, alpha, A, true);
    REQUIRE(C2_add_AH_mul_A.get_property() == LAPACKSupport::Property::general);
    A.Hmmult(C2_add_AH_mul_A_hsymm, alpha, A, true);
    REQUIRE(C2_add_AH_mul_A_hsymm.get_property() ==
            LAPACKSupport::Property::general);
    A.mHmult(C1_add_A_mul_BH, alpha, B, true);
    REQUIRE(C1_add_A_mul_BH.get_property() == LAPACKSupport::Property::general);
    A.Hmmult(C2_add_AH_mul_B, alpha, B, true);
    REQUIRE(C2_add_AH_mul_B.get_property() == LAPACKSupport::Property::general);

    compare_with_jl_matrix(C1_add_A_mul_AH,
                           "C1_add_alpha1_A_mul_AH",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_AH_hsymm,
                           "C1_add_alpha1_A_mul_AH",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AH_mul_A,
                           "C2_add_alpha1_AH_mul_A",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AH_mul_A_hsymm,
                           "C2_add_alpha1_AH_mul_A",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_BH,
                           "C1_add_alpha1_A_mul_BH",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AH_mul_B,
                           "C2_add_alpha1_AH_mul_B",
                           1e-15,
                           1e-15);
  }

  {
    // Generate a Hermite symmetric matrix @p C1_add_A_mul_AH , but set it as a
    // general matrix.
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AH_tmp(m, m);
    for (unsigned int i = 1; i <= m; i++)
      for (unsigned int j = 1; j <= m; j++)
        C1_add_A_mul_AH_tmp(i - 1, j - 1) =
          std::complex<double>(i / 10., j / 10.);
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AH(m, m);
    C1_add_A_mul_AH_tmp.mHmult(C1_add_A_mul_AH, C1_add_A_mul_AH_tmp);
    C1_add_A_mul_AH.set_property(LAPACKSupport::Property::general);

    // Generate a Hermite symmetric matrix @p C1_add_A_mul_AH_hsymm and set its
    // property as it is.
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AH_hsymm(
      C1_add_A_mul_AH);
    C1_add_A_mul_AH_hsymm.set_property(
      LAPACKSupport::Property::hermite_symmetric);

    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_BH(C1_add_A_mul_AH);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_A_tmp(n, n);
    for (unsigned int i = 1; i <= n; i++)
      for (unsigned int j = 1; j <= n; j++)
        C2_add_AH_mul_A_tmp(i - 1, j - 1) =
          std::complex<double>(i / 10., j / 10.);
    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_A(n, n);
    C2_add_AH_mul_A_tmp.mHmult(C2_add_AH_mul_A, C2_add_AH_mul_A_tmp);
    C2_add_AH_mul_A.set_property(LAPACKSupport::Property::general);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_A_hsymm(
      C2_add_AH_mul_A);
    C2_add_AH_mul_A_hsymm.set_property(
      LAPACKSupport::Property::hermite_symmetric);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AH_mul_B(C2_add_AH_mul_A);

    double alpha = 2.5;

    A.mHmult(C1_add_A_mul_AH, alpha, A, true);
    REQUIRE(C1_add_A_mul_AH.get_property() == LAPACKSupport::Property::general);
    A.mHmult(C1_add_A_mul_AH_hsymm, alpha, A, true);
    REQUIRE(C1_add_A_mul_AH_hsymm.get_property() ==
            LAPACKSupport::Property::hermite_symmetric);
    A.Hmmult(C2_add_AH_mul_A, alpha, A, true);
    REQUIRE(C2_add_AH_mul_A.get_property() == LAPACKSupport::Property::general);
    A.Hmmult(C2_add_AH_mul_A_hsymm, alpha, A, true);
    REQUIRE(C2_add_AH_mul_A_hsymm.get_property() ==
            LAPACKSupport::Property::hermite_symmetric);
    A.mHmult(C1_add_A_mul_BH, alpha, B, true);
    REQUIRE(C1_add_A_mul_BH.get_property() == LAPACKSupport::Property::general);
    A.Hmmult(C2_add_AH_mul_B, alpha, B, true);
    REQUIRE(C2_add_AH_mul_B.get_property() == LAPACKSupport::Property::general);

    compare_with_jl_matrix(C1_add_A_mul_AH,
                           "C1_add_alpha2_A_mul_AH",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_AH_hsymm,
                           "C1_add_alpha2_A_mul_AH",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AH_mul_A,
                           "C2_add_alpha2_AH_mul_A",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AH_mul_A_hsymm,
                           "C2_add_alpha2_AH_mul_A",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_BH,
                           "C1_add_alpha2_A_mul_BH",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AH_mul_B,
                           "C2_add_alpha2_AH_mul_B",
                           1e-15,
                           1e-15);
  }

  INFO("*** test end");
}
