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
 * \file lapack-matrix-reduced-svd.cc
 * \brief Verify reduced SVD.
 * \ingroup linalg
 *
 * \author Jihuan Tian
 * \date 2021-07-05
 */

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

TEST_CASE("Verify reduced SVD decomposition for LAPACKFullMatrixExt",
          "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  const unsigned int m = 3;
  const unsigned int n = 5;

  LAPACKFullMatrixExt<double> A_original(m, n);
  LAPACKFullMatrixExt<double> U, VT;
  std::vector<double>         Sigma_r;

  unsigned int counter = 1;
  for (unsigned int i = 0; i < m; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A_original(i, j) = (double)counter;
          counter++;
        }
    }

  A_original.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");

  LAPACKFullMatrixExt<std::complex<double>> A_complex_original(m, n);
  LAPACKFullMatrixExt<std::complex<double>> U_complex, VT_complex;
  std::vector<double>                       Sigma_r_complex;

  for (unsigned int i = 0; i < m; i++)
    for (unsigned int j = 0; j < n; j++)
      A_complex_original(i, j) = {sin(i + 1.0), cos(j + 1.0)};

  A_complex_original.print_formatted_to_mat(std::cout, "Ac", 8, false, 25, "0");

  // Functor to be applied to scalar value for comparison. Because there may be
  // sign differences in U and VT between Julia and C++, we compare their
  // absolute values.
  auto abs_func = [](double v) -> double { return std::abs(v); };

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 1\n";

    A.reduced_svd(U, Sigma_r, VT, 1);

    U.print_formatted_to_mat(std::cout, "U1", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT1", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r1", Sigma_r);

    compare_with_jl_array(U, "U[:,1]", 1e-14, 1e-14, abs_func);
    compare_with_jl_array(VT, "VT[1,:]", 1e-14, 1e-14, abs_func);
    compare_with_jl_scalar(Sigma_r[0], "Sigma_r[1]", 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> A_complex(A_complex_original);

    std::cout << "Complex valued RSVD with rank truncated to 1\n";

    A_complex.reduced_svd(U_complex, Sigma_r_complex, VT_complex, 1);

    U_complex.print_formatted_to_mat(
      std::cout, "U_complex1", 8, false, 25, "0");
    VT_complex.print_formatted_to_mat(
      std::cout, "VT_complex1", 8, false, 25, "0");
    print_vector_to_mat(std::cout, "Sigma_r_complex1", Sigma_r_complex);

    compare_with_jl_array(U_complex, "U_complex[:,1]", 1e-14, 1e-14, abs_func);
    compare_with_jl_array(
      VT_complex, "VT_complex[1,:]", 1e-14, 1e-14, abs_func);
    compare_with_jl_scalar(Sigma_r_complex[0],
                           "Sigma_r_complex[1]",
                           1e-15,
                           1e-15);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 2\n";

    A.reduced_svd(U, Sigma_r, VT, 2);

    U.print_formatted_to_mat(std::cout, "U2", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT2", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r2", Sigma_r);

    compare_with_jl_matrix(U, "U[:,1:2]", 1e-14, 1e-14, abs_func);
    compare_with_jl_matrix(VT, "VT[1:2,:]", 1e-14, 1e-14, abs_func);
    std::vector<double> Sigma_r_tmp{Sigma_r[0], Sigma_r[1]};
    compare_with_jl_array(Sigma_r_tmp, "Sigma_r[1:2]", 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> A_complex(A_complex_original);

    std::cout << "Complex valued RSVD with rank truncated to 2\n";

    A_complex.reduced_svd(U_complex, Sigma_r_complex, VT_complex, 2);

    U_complex.print_formatted_to_mat(
      std::cout, "U_complex2", 8, false, 25, "0");
    VT_complex.print_formatted_to_mat(
      std::cout, "VT_complex2", 8, false, 25, "0");
    print_vector_to_mat(std::cout, "Sigma_r_complex2", Sigma_r_complex);

    compare_with_jl_matrix(
      U_complex, "U_complex[:,1:2]", 1e-14, 1e-14, abs_func);
    compare_with_jl_matrix(
      VT_complex, "VT_complex[1:2,:]", 1e-14, 1e-14, abs_func);
    std::vector<double> Sigma_r_complex_tmp{Sigma_r_complex[0],
                                            Sigma_r_complex[1]};
    compare_with_jl_array(Sigma_r_complex_tmp,
                          "Sigma_r_complex[1:2]",
                          1e-15,
                          1e-15);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 3\n";

    // Due to automatically setting singular value threshold, there are only two
    // columns kept in U and V.
    A.reduced_svd(U, Sigma_r, VT, 3);

    U.print_formatted_to_mat(std::cout, "U3", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT3", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r3", Sigma_r);

    compare_with_jl_matrix(U, "U[:,1:2]", 1e-14, 1e-14, abs_func);
    compare_with_jl_matrix(VT, "VT[1:2,:]", 1e-14, 1e-14, abs_func);
    std::vector<double> Sigma_r_tmp{Sigma_r[0], Sigma_r[1]};
    compare_with_jl_array(Sigma_r_tmp, "Sigma_r[1:2]", 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> A_complex(A_complex_original);

    std::cout << "Complex valued RSVD with rank truncated to 3\n";

    // Due to automatically setting singular value threshold, there are only two
    // columns kept in U and V.
    A_complex.reduced_svd(U_complex, Sigma_r_complex, VT_complex, 3);

    U_complex.print_formatted_to_mat(
      std::cout, "U_complex3", 8, false, 25, "0");
    VT_complex.print_formatted_to_mat(
      std::cout, "VT_complex3", 8, false, 25, "0");
    print_vector_to_mat(std::cout, "Sigma_r_complex3", Sigma_r_complex);

    compare_with_jl_matrix(
      U_complex, "U_complex[:,1:2]", 1e-14, 1e-14, abs_func);
    compare_with_jl_matrix(
      VT_complex, "VT_complex[1:2,:]", 1e-14, 1e-14, abs_func);
    std::vector<double> Sigma_r_complex_tmp{Sigma_r_complex[0],
                                            Sigma_r_complex[1]};
    compare_with_jl_array(Sigma_r_complex_tmp,
                          "Sigma_r_complex[1:2]",
                          1e-15,
                          1e-15);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 4\n";

    // Due to automatically setting singular value threshold, there are only two
    // columns kept in U and V.
    A.reduced_svd(U, Sigma_r, VT, 4);

    U.print_formatted_to_mat(std::cout, "U4", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT4", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r4", Sigma_r);

    compare_with_jl_matrix(U, "U[:,1:2]", 1e-14, 1e-14, abs_func);
    compare_with_jl_matrix(VT, "VT[1:2,:]", 1e-14, 1e-14, abs_func);
    std::vector<double> Sigma_r_tmp{Sigma_r[0], Sigma_r[1]};
    compare_with_jl_array(Sigma_r_tmp, "Sigma_r[1:2]", 1e-15, 1e-15);

    LAPACKFullMatrixExt<std::complex<double>> A_complex(A_complex_original);

    std::cout << "Complex valued RSVD with rank truncated to 4\n";

    // Due to automatically setting singular value threshold, there are only two
    // columns kept in U and V.
    A_complex.reduced_svd(U_complex, Sigma_r_complex, VT_complex, 4);

    U_complex.print_formatted_to_mat(
      std::cout, "U_complex4", 8, false, 25, "0");
    VT_complex.print_formatted_to_mat(
      std::cout, "VT_complex4", 8, false, 25, "0");
    print_vector_to_mat(std::cout, "Sigma_r_complex4", Sigma_r_complex);

    compare_with_jl_matrix(
      U_complex, "U_complex[:,1:2]", 1e-14, 1e-14, abs_func);
    compare_with_jl_matrix(
      VT_complex, "VT_complex[1:2,:]", 1e-14, 1e-14, abs_func);
    std::vector<double> Sigma_r_complex_tmp{Sigma_r_complex[0],
                                            Sigma_r_complex[1]};
    compare_with_jl_array(Sigma_r_complex_tmp,
                          "Sigma_r_complex[1:2]",
                          1e-15,
                          1e-15);
  }

  INFO("*** test end");
}
