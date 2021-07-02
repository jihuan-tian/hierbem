/**
 * \file rkmatrix-add-formatted.cc
 * \brief Verify the formatted addition of two rank-k matrices.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-06-30
 */

#include <iostream>

#include "debug_tools.h"
#include "rkmatrix.h"

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
  M1.print_formatted_to_mat(std::cout, "M1", 5, false, 10, "0");

  std::vector<double> values2{3, 8, 10, 7, 1, 9, 7, 6, 12, 4, 5, 8, 8, 9, 20};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values2, M2);
  M2.print_formatted_to_mat(std::cout, "M2", 5, false, 10, "0");

  /**
   * Create two rank-k matrices converted from the two matrices.
   */
  const unsigned int rank = 3;

  RkMatrix<double> A(rank, M1);
  A.print_formatted_to_mat(std::cout, "A", 5, false, 10, "0");

  RkMatrix<double> B(rank, M2);
  B.print_formatted_to_mat(std::cout, "B", 5, false, 10, "0");

  /**
   * Perform formatted addition.
   */
  {
    RkMatrix<double> C;
    A.add(C, B, 1);
    C.print_formatted_to_mat(std::cout, "C_trunc_1", 5, false, 10, "0");
  }

  {
    RkMatrix<double> C;
    A.add(C, B, 2);
    C.print_formatted_to_mat(std::cout, "C_trunc_2", 5, false, 10, "0");
  }

  {
    RkMatrix<double> C;
    A.add(C, B, 3);
    C.print_formatted_to_mat(std::cout, "C_trunc_3", 5, false, 10, "0");
  }

  return 0;
}
