/**
 * \file lapack-matrix-rk-decompose.cc
 * \brief Verify decomposition of a full matrix into the two components of a
 * rank-k matrix.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-05
 */

#include "lapack_full_matrix_ext.h"

int
main()
{
  /**
   * Construct a full matrix.
   */
  LAPACKFullMatrixExt<double> M_orig;
  std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M_orig);

  M_orig.print_formatted_to_mat(std::cout, "M", 8, true, 16, "0");

  {
    /**
     * Decompose the full matrix into the product of \p A and \p B^T with
     * rank truncated to 1.
     */
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 1;

    M.rank_k_decompose(k, A, B);

    A.print_formatted_to_mat(std::cout, "A1", 8, true, 16, "0");
    B.print_formatted_to_mat(std::cout, "B1", 8, true, 16, "0");
  }

  {
    /**
     * Decompose the full matrix into the product of \p A and \p B^T with
     * rank truncated to 2.
     */
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 2;

    M.rank_k_decompose(k, A, B);

    A.print_formatted_to_mat(std::cout, "A2", 8, true, 16, "0");
    B.print_formatted_to_mat(std::cout, "B2", 8, true, 16, "0");
  }

  {
    /**
     * Decompose the full matrix into the product of \p A and \p B^T with
     * rank truncated to 3.
     */
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 3;

    M.rank_k_decompose(k, A, B);

    A.print_formatted_to_mat(std::cout, "A3", 8, true, 16, "0");
    B.print_formatted_to_mat(std::cout, "B3", 8, true, 16, "0");
  }

  {
    /**
     * Decompose the full matrix into the product of \p A and \p B^T with
     * rank truncated to 4.
     */
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 4;

    M.rank_k_decompose(k, A, B);

    A.print_formatted_to_mat(std::cout, "A4", 8, true, 16, "0");
    B.print_formatted_to_mat(std::cout, "B4", 8, true, 16, "0");
  }

  return 0;
}
