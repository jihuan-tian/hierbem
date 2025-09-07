/**
 * \file lapack-matrix-rsvd-degenerate-cases.cc
 * \brief Verify degenerate cases for reduced SVD.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-05
 */

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int rank = 1;

  {
    /**
     * RSVD of a scalar matrix.
     */
    LAPACKFullMatrixExt<double> A(1, 1);
    LAPACKFullMatrixExt<double> U, VT;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A.print_formatted_to_mat(std::cout, "A1", 8, false, 16, "0");

    A.reduced_svd(U, Sigma_r, VT, rank);

    U.print_formatted_to_mat(std::cout, "U1", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT1", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r1", Sigma_r);
  }

  {
    /**
     * RSVD of a row matrix.
     */
    LAPACKFullMatrixExt<double> A(1, 3);
    LAPACKFullMatrixExt<double> U, VT;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A(0, 1) = 6.;
    A(0, 2) = 3.;
    A.print_formatted_to_mat(std::cout, "A2", 8, false, 16, "0");

    A.reduced_svd(U, Sigma_r, VT, rank);

    U.print_formatted_to_mat(std::cout, "U2", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT2", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r2", Sigma_r);
  }

  {
    /**
     * RSVD of a column matrix.
     */
    LAPACKFullMatrixExt<double> A(3, 1);
    LAPACKFullMatrixExt<double> U, VT;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A(1, 0) = 6.;
    A(2, 0) = 3.;
    A.print_formatted_to_mat(std::cout, "A3", 8, false, 16, "0");

    A.reduced_svd(U, Sigma_r, VT, rank);

    U.print_formatted_to_mat(std::cout, "U3", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT3", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r3", Sigma_r);
  }

  return 0;
}
