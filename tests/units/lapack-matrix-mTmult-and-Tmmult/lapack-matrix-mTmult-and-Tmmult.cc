/**
 * @file lapack-matrix-mTmult-and-Tmmult.cc
 * @brief Verify mTmult and Tmmult for both real and complex valued
 * LAPACKFullMatrixExt
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

#include "debug_tools.h"
#include "hbem_cpp_validate.h"
#include "hbem_julia_cpp_compare.h"
#include "hbem_julia_wrapper.h"
#include "lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE(
  "Verify mTmult and Tmmult for both real and complex valued LAPACKFullMatrixExt",
  "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  // Real valued
  {
    const unsigned int m = 3;
    const unsigned int n = 5;

    double                     *mat_ptr;
    LAPACKFullMatrixExt<double> A(m, n);
    mat_ptr = &(A.begin()->value());
    for (unsigned int i = 1; i <= m * n; i++, mat_ptr++)
      {
        (*mat_ptr) = std::sin((double)i);
      }

    LAPACKFullMatrixExt<double> B(m, n);
    mat_ptr = &(B.begin()->value());
    for (unsigned int i = 1; i <= m * n; i++, mat_ptr++)
      {
        (*mat_ptr) = std::cos((double)i);
      }

    LAPACKFullMatrixExt<double> C1_add_A_mul_AT(m, m);
    for (unsigned int i = 0; i < m; i++)
      for (unsigned int j = 0; j < m; j++)
        C1_add_A_mul_AT(i, j) = (i + 1) * (j + 1);

    LAPACKFullMatrixExt<double> C1_add_A_mul_AT_symm(C1_add_A_mul_AT);
    C1_add_A_mul_AT_symm.set_property(LAPACKSupport::Property::symmetric);
    LAPACKFullMatrixExt<double> C1_add_A_mul_BT(C1_add_A_mul_AT);

    LAPACKFullMatrixExt<double> C2_add_AT_mul_A(n, n);
    for (unsigned int i = 0; i < n; i++)
      for (unsigned int j = 0; j < n; j++)
        C2_add_AT_mul_A(i, j) = (i + 1) * (j + 1);

    LAPACKFullMatrixExt<double> C2_add_AT_mul_A_symm(C2_add_AT_mul_A);
    C2_add_AT_mul_A_symm.set_property(LAPACKSupport::Property::symmetric);
    LAPACKFullMatrixExt<double> C2_add_AT_mul_B(C2_add_AT_mul_A);

    double alpha = 2.5;

    LAPACKFullMatrixExt<double> A_mul_AT(m, m);
    LAPACKFullMatrixExt<double> AT_mul_A(n, n);
    LAPACKFullMatrixExt<double> A_mul_BT(m, m);
    LAPACKFullMatrixExt<double> AT_mul_B(n, n);

    // Multiplication without adding into the result matrix.
    A.mTmult(A_mul_AT, A);
    REQUIRE(A_mul_AT.get_property() == LAPACKSupport::Property::symmetric);
    A.Tmmult(AT_mul_A, A);
    REQUIRE(AT_mul_A.get_property() == LAPACKSupport::Property::symmetric);
    A.mTmult(A_mul_BT, B);
    REQUIRE(A_mul_BT.get_property() == LAPACKSupport::Property::general);
    A.Tmmult(AT_mul_B, B);
    REQUIRE(AT_mul_B.get_property() == LAPACKSupport::Property::general);

    compare_with_jl_matrix(A_mul_AT, "A_mul_AT_real", 1e-15, 1e-15);
    compare_with_jl_matrix(AT_mul_A, "AT_mul_A_real", 1e-15, 1e-15);
    compare_with_jl_matrix(A_mul_BT, "A_mul_BT_real", 1e-15, 1e-15);
    compare_with_jl_matrix(AT_mul_B, "AT_mul_B_real", 1e-15, 1e-15);

    // Multiplication with adding into the result matrix.
    A.mTmult(C1_add_A_mul_AT, alpha, A, true);
    REQUIRE(C1_add_A_mul_AT.get_property() == LAPACKSupport::Property::general);
    A.mTmult(C1_add_A_mul_AT_symm, alpha, A, true);
    REQUIRE(C1_add_A_mul_AT_symm.get_property() ==
            LAPACKSupport::Property::symmetric);
    A.Tmmult(C2_add_AT_mul_A, alpha, A, true);
    REQUIRE(C2_add_AT_mul_A.get_property() == LAPACKSupport::Property::general);
    A.Tmmult(C2_add_AT_mul_A_symm, alpha, A, true);
    REQUIRE(C2_add_AT_mul_A_symm.get_property() ==
            LAPACKSupport::Property::symmetric);
    A.mTmult(C1_add_A_mul_BT, alpha, B, true);
    REQUIRE(C1_add_A_mul_BT.get_property() == LAPACKSupport::Property::general);
    A.Tmmult(C2_add_AT_mul_B, alpha, B, true);
    REQUIRE(C2_add_AT_mul_B.get_property() == LAPACKSupport::Property::general);

    compare_with_jl_matrix(C1_add_A_mul_AT,
                           "C1_add_A_mul_AT_real",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_AT_symm,
                           "C1_add_A_mul_AT_real",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AT_mul_A,
                           "C2_add_AT_mul_A_real",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AT_mul_A_symm,
                           "C2_add_AT_mul_A_real",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_BT,
                           "C1_add_A_mul_BT_real",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AT_mul_B,
                           "C2_add_AT_mul_B_real",
                           1e-15,
                           1e-15);
  }

  // Complex valued
  {
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

    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AT(m, m);
    for (unsigned int i = 1; i <= m; i++)
      for (unsigned int j = 1; j <= m; j++)
        C1_add_A_mul_AT(i - 1, j - 1) = std::complex<double>(i * j, i * j);

    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_AT_symm(
      C1_add_A_mul_AT);
    C1_add_A_mul_AT_symm.set_property(LAPACKSupport::Property::symmetric);
    LAPACKFullMatrixExt<std::complex<double>> C1_add_A_mul_BT(C1_add_A_mul_AT);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AT_mul_A(n, n);
    for (unsigned int i = 1; i <= n; i++)
      for (unsigned int j = 1; j <= n; j++)
        C2_add_AT_mul_A(i - 1, j - 1) = std::complex<double>(i * j, i * j);

    LAPACKFullMatrixExt<std::complex<double>> C2_add_AT_mul_A_symm(
      C2_add_AT_mul_A);
    C2_add_AT_mul_A_symm.set_property(LAPACKSupport::Property::symmetric);
    LAPACKFullMatrixExt<std::complex<double>> C2_add_AT_mul_B(C2_add_AT_mul_A);

    std::complex<double> alpha(2.5, 1.2);

    LAPACKFullMatrixExt<std::complex<double>> A_mul_AT(m, m);
    LAPACKFullMatrixExt<std::complex<double>> AT_mul_A(n, n);
    LAPACKFullMatrixExt<std::complex<double>> A_mul_BT(m, m);
    LAPACKFullMatrixExt<std::complex<double>> AT_mul_B(n, n);

    // Multiplication without adding into the result matrix.
    A.mTmult(A_mul_AT, A);
    REQUIRE(A_mul_AT.get_property() == LAPACKSupport::Property::symmetric);
    A.Tmmult(AT_mul_A, A);
    REQUIRE(AT_mul_A.get_property() == LAPACKSupport::Property::symmetric);
    A.mTmult(A_mul_BT, B);
    REQUIRE(A_mul_BT.get_property() == LAPACKSupport::Property::general);
    A.Tmmult(AT_mul_B, B);
    REQUIRE(AT_mul_B.get_property() == LAPACKSupport::Property::general);

    compare_with_jl_matrix(A_mul_AT, "A_mul_AT_complex", 1e-15, 1e-15);
    compare_with_jl_matrix(AT_mul_A, "AT_mul_A_complex", 1e-15, 1e-15);
    compare_with_jl_matrix(A_mul_BT, "A_mul_BT_complex", 1e-15, 1e-15);
    compare_with_jl_matrix(AT_mul_B, "AT_mul_B_complex", 1e-15, 1e-15);

    // Multiplication with adding into the result matrix.
    A.mTmult(C1_add_A_mul_AT, alpha, A, true);
    REQUIRE(C1_add_A_mul_AT.get_property() == LAPACKSupport::Property::general);
    A.mTmult(C1_add_A_mul_AT_symm, alpha, A, true);
    REQUIRE(C1_add_A_mul_AT_symm.get_property() ==
            LAPACKSupport::Property::symmetric);
    A.Tmmult(C2_add_AT_mul_A, alpha, A, true);
    REQUIRE(C2_add_AT_mul_A.get_property() == LAPACKSupport::Property::general);
    A.Tmmult(C2_add_AT_mul_A_symm, alpha, A, true);
    REQUIRE(C2_add_AT_mul_A_symm.get_property() ==
            LAPACKSupport::Property::symmetric);
    A.mTmult(C1_add_A_mul_BT, alpha, B, true);
    REQUIRE(C1_add_A_mul_BT.get_property() == LAPACKSupport::Property::general);
    A.Tmmult(C2_add_AT_mul_B, alpha, B, true);
    REQUIRE(C2_add_AT_mul_B.get_property() == LAPACKSupport::Property::general);

    compare_with_jl_matrix(C1_add_A_mul_AT,
                           "C1_add_A_mul_AT_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_AT_symm,
                           "C1_add_A_mul_AT_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AT_mul_A,
                           "C2_add_AT_mul_A_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AT_mul_A_symm,
                           "C2_add_AT_mul_A_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C1_add_A_mul_BT,
                           "C1_add_A_mul_BT_complex",
                           1e-15,
                           1e-15);
    compare_with_jl_matrix(C2_add_AT_mul_B,
                           "C2_add_AT_mul_B_complex",
                           1e-15,
                           1e-15);
  }

  INFO("*** test end");
}
