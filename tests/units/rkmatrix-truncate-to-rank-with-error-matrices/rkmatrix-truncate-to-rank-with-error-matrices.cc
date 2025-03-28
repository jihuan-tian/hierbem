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
#include <fstream>
#include <iostream>
#include <vector>

#include "debug_tools.h"
#include "hbem_cpp_validate.h"
#include "lapack_full_matrix_ext.h"
#include "rkmatrix.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE(
  "Verify truncation of an RkMatrix to a given rank with error matrices",
  "[linalg]")
{
  INFO("*** test start");

  std::ofstream ofs("rkmatrix-truncate-to-rank-with-error-matrices.log");

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
      ofs << "=== Truncation rank=" << r << std::endl;

      LAPACKFullMatrixExt<double>               A_copy(A);
      LAPACKFullMatrixExt<std::complex<double>> A_complex_copy(A_complex);

      ofs << "*** Real valued matrix" << std::endl;
      LAPACKFullMatrixExt<double> C, D;
      RkMatrix<double>            A_rk(r, A_copy, C, D);
      A_rk.print_formatted(ofs, 8, false, 15, "0");
      if (C.m() > 0 && C.n() > 0)
        C.print_formatted_to_mat(ofs, "C", 8, false, 25, "0");
      if (D.m() > 0 && D.n() > 0)
        D.print_formatted_to_mat(ofs, "D", 8, false, 25, "0");

      ofs << "*** Complex valued matrix" << std::endl;
      LAPACKFullMatrixExt<std::complex<double>> C_complex, D_complex;
      RkMatrix<std::complex<double>>            A_complex_rk(r,
                                                  A_complex_copy,
                                                  C_complex,
                                                  D_complex);
      A_complex_rk.print_formatted(ofs, 8, false, 25, "0");
      if (C_complex.m() > 0 && C_complex.n() > 0)
        C_complex.print_formatted_to_mat(ofs, "C_complex", 8, false, 25, "0");
      if (D_complex.m() > 0 && D_complex.n() > 0)
        D_complex.print_formatted_to_mat(ofs, "D_complex", 8, false, 25, "0");
    }

  ofs.close();

  auto check_equality = [](const auto &a, const auto &b) {
    INFO("Operand 1: " << a);
    INFO("Operand 2: " << b);
    REQUIRE(a == b);
  };
  compare_two_files(SOURCE_DIR "/reference.output",
                    "rkmatrix-truncate-to-rank-with-error-matrices.log",
                    check_equality);
}
