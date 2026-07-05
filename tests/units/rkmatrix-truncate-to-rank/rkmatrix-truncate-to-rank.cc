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

#include "hbem_cpp_validate.h"
#include "hbem_julia_cpp_compare.h"
#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace Catch::Matchers;
using namespace HierBEM;

TEST_CASE("Verify truncation of an RkMatrix to a given rank", "[linalg]")
{
  INFO("*** test start");

  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file(SOURCE_DIR "/process.jl");

  std::ofstream ofs("rkmatrix-truncate-to-rank.log");

  const unsigned int n = 6;

  // Both full matrices have rank=2.
  const unsigned int                        actual_rank = 2;
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
      A_rk.print_formatted(ofs, 15, false, 25, "0");

      ofs << "*** Complex valued matrix" << std::endl;
      RkMatrix<std::complex<double>> A_complex_rk(A_complex_copy);
      A_complex_rk.truncate_to_rank(r);
      A_complex_rk.print_formatted(ofs, 15, false, 25, "0");

      // Because the signs of the component matrices A and B are indefinite,
      // which depend on the implementation of LAPACK, we do not check A and B
      // themselves, but their outer product. When the truncation rank r >=
      // actual_rank, we compare the product with the original full matrix. When
      // r == 1, we compare the product with the precomputed results.
      LAPACKFullMatrixExt<double>               A_rk_full;
      LAPACKFullMatrixExt<std::complex<double>> A_complex_rk_full;

      A_rk.convertToFullMatrix(A_rk_full);
      A_complex_rk.convertToFullMatrix(A_complex_rk_full);

      if (r >= actual_rank)
        {
          compare_lapack_matrices(A, A_rk_full, 1e-12, 1e-12);
          compare_lapack_matrices(A_complex, A_complex_rk_full, 1e-12, 1e-12);
        }

      if (r == 1)
        {
          compare_with_jl_matrix(A_rk_full, "A_rk_full_jl", 1e-14, 1e-14);
          compare_with_jl_matrix(A_complex_rk_full,
                                 "A_complex_rk_full_jl",
                                 1e-14,
                                 1e-14);
        }
    }

  ofs.close();
}
