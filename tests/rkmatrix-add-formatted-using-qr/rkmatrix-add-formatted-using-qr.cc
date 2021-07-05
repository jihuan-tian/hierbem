/**
 * \file rkmatrix-add-formatted-using-qr.cc
 * \brief Verify the formatted addition of two rank-k matrices by using the QR
 * decomposition. This requires that the component matrices of rank-k matrices
 * should wide matrix, i.e. having more rows than columns.
 *
 * N.B. Because the data of the two operand matrices are created by hand, it
 * cannot ensure a exponential decrease of singular values like that generated
 * from a integral operator. Hence, the approximation accuracy is not high.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-07-02
 */

#include <iostream>
#include <string>

#include "debug_tools.h"
#include "rkmatrix.h"

int
main()
{
  /**
   * Create two full matrices as the data source.
   */
  LAPACKFullMatrixExt<double> M1, M2;

  std::vector<double> values1{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 1,  2,  3,  4,  5,  6,
                              7,  8,  9,  10, 11, 12, 13, 14, 15, 1,  2,  3,
                              4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                              1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 1,  2,  3,  4,  5,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15, 1,  2,  3,  4,  5,  6,
                              7,  8,  9,  10, 11, 12, 13, 14, 15, 1,  2,  3,
                              4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(12, 10, values1, M1);
  M1.print_formatted_to_mat(std::cout, "M1", 8, false, 16, "0");

  std::vector<double> values2{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20,
                              3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};
  LAPACKFullMatrixExt<double>::Reshape(12, 10, values2, M2);
  M2.print_formatted_to_mat(std::cout, "M2", 8, false, 16, "0");

  /**
   * Create two rank-k matrices converted from the two matrices.
   */
  const unsigned int rank = 4;

  RkMatrix<double> A(rank, M1);
  A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");

  RkMatrix<double> B(rank, M2);
  B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");

  /**
   * Perform formatted addition.
   */
  for (unsigned int i = 1; i <= rank; i++)
    {
      RkMatrix<double> C;
      A.add(C, B, i);
      C.print_formatted_to_mat(std::cout,
                               std::string("C_trunc_") + std::to_string(i),
                               8,
                               false,
                               16,
                               "0");
    }

  return 0;
}
