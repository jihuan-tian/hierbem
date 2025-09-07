/**
 * \file lapack-matrix-mmult.cc
 * \brief Verify the multiplication of two \p LAPACKFullMatrixExt.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-08-19
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> a{1, 3, 5, 7, 9, 10};
  std::vector<double> b{2, 8, 9, 7, 1, 3, 11, 20, 13};
  std::vector<double> c{1, 1, 1, 2, 2, 2};

  LAPACKFullMatrixExt<double> A, B, C, C_adding;

  LAPACKFullMatrixExt<double>::Reshape(2, 3, a, A);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, b, B);
  LAPACKFullMatrixExt<double>::Reshape(2, 3, c, C_adding);

  A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");
  B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");
  C_adding.print_formatted_to_mat(
    std::cout, "C_adding_before", 8, false, 16, "0");

  A.mmult(C, B);
  A.mmult(C_adding, B, true);

  C.print_formatted_to_mat(std::cout, "C", 8, false, 16, "0");
  C_adding.print_formatted_to_mat(std::cout, "C_adding", 8, false, 16, "0");
}
