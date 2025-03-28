/**
 * \file rkmatrix-truncate-to-rank.cc
 * \brief Verify the truncation of an RkMatrix to a given rank.
 * \ingroup rkmatrices
 *
 * \author Jihuan Tian
 * \date 2025-03-25
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

TEST_CASE("Verify truncation of an RkMatrix to a given rank", "[linalg]")
{
  INFO("*** test start");

  std::ofstream ofs("rkmatrix-truncate-to-rank.log");

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
      RkMatrix<double> A_rk(A_copy);
      A_rk.truncate_to_rank(r);
      A_rk.print_formatted(ofs, 8, false, 15, "0");

      ofs << "*** Complex valued matrix" << std::endl;
      RkMatrix<std::complex<double>> A_complex_rk(A_complex_copy);
      A_complex_rk.truncate_to_rank(r);
      A_complex_rk.print_formatted(ofs, 8, false, 25, "0");
    }

  ofs.close();

  auto check_equality = [](const auto &a, const auto &b) {
    INFO("Operand 1: " << a);
    INFO("Operand 2: " << b);
    REQUIRE(a == b);
  };
  compare_two_files(SOURCE_DIR "/reference.output",
                    "rkmatrix-truncate-to-rank.log",
                    check_equality);
}
