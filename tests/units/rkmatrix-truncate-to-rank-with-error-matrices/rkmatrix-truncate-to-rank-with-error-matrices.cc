/**
 * \file rkmatrix-truncate-to-rank-with-error-matrices.cc
 * \brief Verify the truncation of an RkMatrix to a given rank with the error
 * matrix returned.
 *
 * \ingroup testers rkmatrices
 * \author Jihuan Tian
 * \date 2021-11-25
 */

#include "debug_tools.h"
#include "rkmatrix.h"

int
main()
{
  /**
   * Create a full matrix with the effective rank being 2, which is not of full
   * rank.
   */
  LAPACKFullMatrixExt<double> M;
  std::vector<double>         values{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  LAPACKFullMatrixExt<double>::Reshape(4, 5, values, M);
  M.print_formatted_to_mat(std::cout, "M");

  /**
   * Convert the full matrix into a rank-3 matrix. Even though rank 3 is
   * required, because it is larger than the effective rank of the full matrix,
   * the rank-k matrix will actually have rank 2, both the formal rank and its
   * actual rank.
   */
  LAPACKFullMatrixExt<double> C, D;
  RkMatrix<double>            A(3, M, C, D);
  A.print_formatted_to_mat(std::cout, "A", 15, false, 25, "0");
  if (C.m() > 0 && C.n() > 0)
    {
      C.print_formatted_to_mat(std::cout, "C", 15, false, 25, "0");
    }

  if (D.m() > 0 && D.n() > 0)
    {
      D.print_formatted_to_mat(std::cout, "D", 15, false, 25, "0");
    }

  {
    /**
     * Truncate the RkMatrix \p A to rank-3. Because the original rank of \p M
     * is 2 and the created rank-k matrix \p A has a rank 2, no actual rank
     * truncation will be performed here.
     */
    LAPACKFullMatrixExt<double> C, D;
    RkMatrix<double>            A_trunc(A);
    A_trunc.truncate_to_rank(3, C, D);
    A_trunc.print_formatted_to_mat(
      std::cout, "A_trunc_to_3", 15, false, 25, "0");
    if (C.m() > 0 && C.n() > 0)
      {
        C.print_formatted_to_mat(std::cout, "C_trunc_to_3", 15, false, 25, "0");
      }

    if (D.m() > 0 && D.n() > 0)
      {
        D.print_formatted_to_mat(std::cout, "D_trunc_to_3", 15, false, 25, "0");
      }
  }

  {
    /**
     * Truncate the RkMatrix \p A to rank-2. Because the original rank of \p M
     * is 2 and the created rank-k matrix \p A has a rank 2, no actual rank
     * truncation will be performed here.
     */
    LAPACKFullMatrixExt<double> C, D;
    RkMatrix<double>            A_trunc(A);
    A_trunc.truncate_to_rank(2, C, D);
    A_trunc.print_formatted_to_mat(std::cout, "A_trunc_to_2");
    if (C.m() > 0 && C.n() > 0)
      {
        C.print_formatted_to_mat(std::cout, "C_trunc_to_2", 15, false, 25, "0");
      }

    if (D.m() > 0 && D.n() > 0)
      {
        D.print_formatted_to_mat(std::cout, "D_trunc_to_2", 15, false, 25, "0");
      }
  }

  {
    /**
     * Truncate the RkMatrix \p A to rank-1. Because the original rank of \p M
     * is 2 and the created rank-k matrix \p A has a rank 2, rank truncation
     * will be performed.
     */
    LAPACKFullMatrixExt<double> C, D;
    RkMatrix<double>            A_trunc(A);
    A_trunc.truncate_to_rank(1, C, D);
    A_trunc.print_formatted_to_mat(std::cout, "A_trunc_to_1");
    if (C.m() > 0 && C.n() > 0)
      {
        C.print_formatted_to_mat(std::cout, "C_trunc_to_1", 15, false, 25, "0");
      }

    if (D.m() > 0 && D.n() > 0)
      {
        D.print_formatted_to_mat(std::cout, "D_trunc_to_1", 15, false, 25, "0");
      }
  }

  return 0;
}
