/**
 * \file svd-degenerate-cases.cc
 * \brief Test SVD and RSVD for degenerate cases, such as the matrix is a
 * scalar, row vector or column vector.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-24
 */

#include "debug_tools.h"
#include "lapack_full_matrix_ext.h"

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

    A.svd(U, Sigma_r, VT);

    print_matrix_to_mat(std::cout, "U", U);
    print_matrix_to_mat(std::cout, "VT", VT);
    print_vector_to_mat(std::cout, "Sigma_r", Sigma_r);
  }

  {
    /**
     * RSVD of a scalar matrix.
     */
    LAPACKFullMatrixExt<double> A(1, 1);
    LAPACKFullMatrixExt<double> B, C;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;

    A.rank_k_decompose(1, B, C);
    print_matrix_to_mat(std::cout, "B", B);
    print_matrix_to_mat(std::cout, "C", C);
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

    A.svd(U, Sigma_r, VT);

    print_matrix_to_mat(std::cout, "U", U);
    print_matrix_to_mat(std::cout, "VT", VT);
    print_vector_to_mat(std::cout, "Sigma_r", Sigma_r);
  }

  {
    /**
     * RSVD of a row matrix.
     */
    LAPACKFullMatrixExt<double> A(1, 3);
    LAPACKFullMatrixExt<double> B, C;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A(0, 1) = 6.;
    A(0, 2) = 3.;

    A.rank_k_decompose(1, B, C);
    print_matrix_to_mat(std::cout, "B", B);
    print_matrix_to_mat(std::cout, "C", C);
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

    A.svd(U, Sigma_r, VT);

    print_matrix_to_mat(std::cout, "U", U);
    print_matrix_to_mat(std::cout, "VT", VT);
    print_vector_to_mat(std::cout, "Sigma_r", Sigma_r);
  }

  {
    /**
     * RSVD of a column matrix.
     */
    LAPACKFullMatrixExt<double> A(3, 1);
    LAPACKFullMatrixExt<double> B, C;
    std::vector<double>         Sigma_r;

    A(0, 0) = 5.;
    A(1, 0) = 6.;
    A(2, 0) = 3.;

    A.rank_k_decompose(1, B, C);
    print_matrix_to_mat(std::cout, "B", B);
    print_matrix_to_mat(std::cout, "C", C);
  }

  return 0;
}
