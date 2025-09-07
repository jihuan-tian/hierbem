/**
 * \file rkmatrix-add-formatted-with-factor.cc
 *
 * \brief Verify the formatted addition of two rank-k matrices \f$C = A + b
 * B\f$.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-05
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "utilities/debug_tools.h"

int
main()
{
  /**
   * Create two full matrices as the data source.
   */
  LAPACKFullMatrixExt<double> M1, M2;

  std::vector<double> values1{
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values1, M1);
  M1.print_formatted_to_mat(std::cout, "M1", 8, false, 16, "0");

  std::vector<double> values2{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values2, M2);
  M2.print_formatted_to_mat(std::cout, "M2", 8, false, 16, "0");

  /**
   * Create two rank-k matrices converted from the two matrices.
   * N.B. The matrix \p M1 has a dimension \f$3 \times 5\f$ but has a rank 2.
   * Even though the rank-k matrix \p A created from \p M1 is declared to have
   * rank 3, the final actual rank is automatically truncated to 2.
   */
  const unsigned int rank = 3;

  RkMatrix<double> A(rank, M1);
  A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");

  RkMatrix<double> B(rank, M2);
  B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");

  /**
   * Perform raw addition of rank-k matrices via juxtaposition.
   */
  double b = 3.5;
  {
    RkMatrix<double> C;
    A.add(C, b, B);
    C.print_formatted_to_mat(std::cout, "C", 8, false, 16, "0");
  }

  /**
   * Perform formatted addition with different truncation ranks.
   */
  {
    RkMatrix<double> C;
    A.add(C, b, B, 1);
    C.print_formatted_to_mat(std::cout, "C_trunc_1", 8, false, 16, "0");
  }

  {
    RkMatrix<double> C;
    A.add(C, b, B, 2);
    C.print_formatted_to_mat(std::cout, "C_trunc_2", 8, false, 16, "0");
  }

  {
    RkMatrix<double> C;
    A.add(C, b, B, 3);
    C.print_formatted_to_mat(std::cout, "C_trunc_3", 8, false, 16, "0");
  }

  /**
   * Calculate \f$A = A + b B\f$.
   */
  {
    A.add(b, B);
    A.print_formatted_to_mat(std::cout, "A_plus_B", 8, false, 16, "0");
  }

  return 0;
}
