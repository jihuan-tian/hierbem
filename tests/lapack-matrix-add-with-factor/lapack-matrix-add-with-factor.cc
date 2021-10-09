/**
 * \file lapack-matrix-add-with-factor.cc
 * \brief Verify matrix addition \f$C = A + b B\f$
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-10-05
 */

#include <cmath>
#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> A_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B_data{3, 5, 7, 4, 6, 8, 5, 7, 9};

  LAPACKFullMatrixExt<double> A, B;
  LAPACKFullMatrixExt<double>::Reshape(3, 3, A_data, A);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, B_data, B);

  A.print_formatted_to_mat(std::cout, "A", 8, false, 12, "0");
  B.print_formatted_to_mat(std::cout, "B", 8, false, 12, "0");

  LAPACKFullMatrixExt<double> C;
  double                      b = 3.5;
  A.add(C, b, B);
  C.print_formatted_to_mat(std::cout, "C", 8, false, 12, "0");

  A.add(b, B);
  A.print_formatted_to_mat(std::cout, "A_self_added", 8, false, 12, "0");

  return 0;
}
