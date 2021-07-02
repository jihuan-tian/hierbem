/**
 * \file rkmatrix-truncate-to-rank.cc
 * \brief Verify the truncation of an RkMatrix to a given rank.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-06-24
 */

#include "debug_tools.h"
#include "rkmatrix.h"

int
main()
{
  /**
   * Create a full matrix.
   */
  LAPACKFullMatrixExt<double> M;
  std::vector<double>         values{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  LAPACKFullMatrixExt<double>::Reshape(4, 5, values, M);
  M.print_formatted_to_mat(std::cout, "M");

  /**
   * Convert the full matrix into a rank-3 matrix.
   */
  RkMatrix<double> A(3, M);
  A.print_formatted_to_mat(std::cout, "A");

  {
    /**
     * Truncate the RkMatrix \p A to rank-3. Because the original rank of \p A
     * is 3, no actual work will be done here.
     */
    RkMatrix<double> A_trunc(A);
    A_trunc.truncate_to_rank(3);
    A_trunc.print_formatted_to_mat(std::cout, "A_trunc_to_3");
  }

  {
    /**
     * Truncate the RkMatrix \p A to rank-2.
     */
    RkMatrix<double> A_trunc(A);
    A_trunc.truncate_to_rank(2);
    A_trunc.print_formatted_to_mat(std::cout, "A_trunc_to_2");
  }

  {
    /**
     * Truncate the RkMatrix \p A to rank-1.
     */
    RkMatrix<double> A_trunc(A);
    A_trunc.truncate_to_rank(1);
    A_trunc.print_formatted_to_mat(std::cout, "A_trunc_to_1");
  }

  return 0;
}
