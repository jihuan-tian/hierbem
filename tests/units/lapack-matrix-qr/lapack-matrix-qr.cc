/**
 * \file lapack-matrix-qr.cc
 * \brief Verify QR decomposition.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-02
 */

#include "lapack_full_matrix_ext.h"

int
main()
{
  LAPACKFullMatrixExt<double> M, Q, R;
  std::vector<double> values{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};

  {
    /**
     * QR decomposition of a matrix with more columns than rows.
     */
    LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M);
    M.print_formatted_to_mat(std::cout, "M1", 8, false, 16, "0");

    M.qr(Q, R);
    Q.print_formatted_to_mat(std::cout, "Q1", 8, false, 16, "0");
    R.print_formatted_to_mat(std::cout, "R1", 8, false, 16, "0");
  }

  {
    /**
     * QR decomposition of a matrix with more rows than columns.
     */
    LAPACKFullMatrixExt<double>::Reshape(5, 3, values, M);
    M.print_formatted_to_mat(std::cout, "M2", 8, false, 16, "0");

    M.qr(Q, R);
    Q.print_formatted_to_mat(std::cout, "Q2", 8, false, 16, "0");
    R.print_formatted_to_mat(std::cout, "R2", 8, false, 16, "0");
  }

  {
    /**
     * Reduced QR decomposition of a matrix with more rows than columns.
     */
    LAPACKFullMatrixExt<double>::Reshape(5, 3, values, M);
    M.print_formatted_to_mat(std::cout, "M3", 8, false, 16, "0");

    M.reduced_qr(Q, R);
    Q.print_formatted_to_mat(std::cout, "Q3", 8, false, 16, "0");
    R.print_formatted_to_mat(std::cout, "R3", 8, false, 16, "0");
  }

  return 0;
}
