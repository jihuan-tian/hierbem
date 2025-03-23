/**
 * \file lapack-matrix-rank-k-decompose.cc
 * \brief Verify decomposition of a full matrix into the two components of a
 * rank-k matrix.
 * \ingroup linalg
 *
 * \author Jihuan Tian
 * \date 2021-07-05
 */

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "debug_tools.h"
#include "hbem_cpp_validate.h"
#include "lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

TEST_CASE("Verify reduced SVD decomposition for LAPACKFullMatrixExt",
          "[linalg]")
{
  INFO("*** test start");

  const unsigned int m = 3;
  const unsigned int n = 5;

  LAPACKFullMatrixExt<double> M_original(m, n);

  unsigned int counter = 1;
  for (unsigned int i = 0; i < m; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          M_original(i, j) = (double)counter;
          counter++;
        }
    }

  M_original.print_formatted_to_mat(std::cout, "M", 8, false, 16, "0");

  LAPACKFullMatrixExt<std::complex<double>> M_complex_original(m, n);
  LAPACKFullMatrixExt<std::complex<double>> A_complex, B_complex;

  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      M_complex_original(i, j) = {sin(i + 1.0), cos(j + 1.0)};

  M_complex_original.print_formatted_to_mat(std::cout, "Mc", 8, false, 25, "0");

  // We already know that the either the real or complex valued matrix have rank
  // 2, so we select the truncate rank to be >=2, so that the product of A and
  // B^H should be exactly the same as the original matrix, then we can check
  // the equality.
  {
    LAPACKFullMatrixExt<double> M(M_original);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 2;

    std::cout << "Rank-2 decomposition\n";

    M.rank_k_decompose(k, A, B);

    A.print_formatted_to_mat(std::cout, "A2", 8, true, 16, "0");
    B.print_formatted_to_mat(std::cout, "B2", 8, true, 16, "0");

    LAPACKFullMatrixExt<double> M_tmp;
    A.mTmult(M_tmp, B);
    compare_lapack_matrices(M_original, M_tmp, 1e-15, 1e-15);
    A.mHmult(M_tmp, B);
    compare_lapack_matrices(M_original, M_tmp, 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> M_complex(M_complex_original);
    LAPACKFullMatrixExt<std::complex<double>> A_complex, B_complex;

    std::cout << "Complex valued rank-2 decomposition\n";

    M_complex.rank_k_decompose(k, A_complex, B_complex);

    A_complex.print_formatted_to_mat(std::cout, "A_complex2", 8, true, 25, "0");
    B_complex.print_formatted_to_mat(std::cout, "B_complex2", 8, true, 25, "0");

    LAPACKFullMatrixExt<std::complex<double>> M_complex_tmp;
    A_complex.mHmult(M_complex_tmp, B_complex);
    compare_lapack_matrices(M_complex_original, M_complex_tmp, 1e-15, 1e-15);
  }

  {
    LAPACKFullMatrixExt<double> M(M_original);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 3;

    std::cout << "Rank-3 decomposition\n";

    M.rank_k_decompose(k, A, B);

    A.print_formatted_to_mat(std::cout, "A3", 8, true, 16, "0");
    B.print_formatted_to_mat(std::cout, "B3", 8, true, 16, "0");

    LAPACKFullMatrixExt<double> M_tmp;
    A.mTmult(M_tmp, B);
    compare_lapack_matrices(M_original, M_tmp, 1e-15, 1e-15);
    A.mHmult(M_tmp, B);
    compare_lapack_matrices(M_original, M_tmp, 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> M_complex(M_complex_original);
    LAPACKFullMatrixExt<std::complex<double>> A_complex, B_complex;

    std::cout << "Complex valued rank-3 decomposition\n";

    M_complex.rank_k_decompose(k, A_complex, B_complex);

    A_complex.print_formatted_to_mat(std::cout, "A_complex3", 8, true, 25, "0");
    B_complex.print_formatted_to_mat(std::cout, "B_complex3", 8, true, 25, "0");

    LAPACKFullMatrixExt<std::complex<double>> M_complex_tmp;
    A_complex.mHmult(M_complex_tmp, B_complex);
    compare_lapack_matrices(M_complex_original, M_complex_tmp, 1e-15, 1e-15);
  }

  INFO("*** test end");
}
