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
 * \file svd-degenerate-cases.cc
 * \brief Test SVD and RSVD for degenerate cases, such as the matrix is a
 * scalar, row vector or column vector.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-24
 */

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  {
    /**
     * SVD of a scalar matrix.
     */
    LAPACKFullMatrixExt<double> A(1, 1);
    LAPACKFullMatrixExt<double> U, VT;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A.print_formatted_to_mat(std::cout, "A1", 8, false, 16, "0");

    A.svd(U, Sigma_r, VT);

    U.print_formatted_to_mat(std::cout, "U1", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT1", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r1", Sigma_r);
  }

  {
    /**
     * SVD of a row matrix.
     */
    LAPACKFullMatrixExt<double> A(1, 3);
    LAPACKFullMatrixExt<double> U, VT;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A(0, 1) = 6.;
    A(0, 2) = 3.;
    A.print_formatted_to_mat(std::cout, "A2", 8, false, 16, "0");

    A.svd(U, Sigma_r, VT);

    U.print_formatted_to_mat(std::cout, "U2", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT2", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r2", Sigma_r);
  }

  {
    /**
     * SVD of a column matrix.
     */
    LAPACKFullMatrixExt<double> A(3, 1);
    LAPACKFullMatrixExt<double> U, VT;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A(1, 0) = 6.;
    A(2, 0) = 3.;
    A.print_formatted_to_mat(std::cout, "A3", 8, false, 16, "0");

    A.svd(U, Sigma_r, VT);

    U.print_formatted_to_mat(std::cout, "U3", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT3", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r3", Sigma_r);
  }

  return 0;
}
