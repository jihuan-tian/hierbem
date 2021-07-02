/**
 * \file hstack-vstack.cc
 * \brief Verify horizontal and vertical stacking of two LAPACKFullMatrixExt
 * objects.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-30
 */

#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  LAPACKFullMatrixExt<double> A, B, C;
  std::vector<double>         A_data{1, 2, 3, 4, 5, 6};
  std::vector<double>         B_data{7, 8, 9, 10, 11, 12};

  LAPACKFullMatrixExt<double>::Reshape(2, 3, A_data, A);
  LAPACKFullMatrixExt<double>::Reshape(2, 3, B_data, B);

  std::cout << "A=\n";
  A.print_formatted(std::cout);

  std::cout << "B=\n";
  B.print_formatted(std::cout);

  std::cout << "Horizontal stacking [A, B]=: \n";
  A.hstack(C, B);
  C.print_formatted(std::cout);

  std::cout << "Vertical stacking [A; B]=: \n";
  A.vstack(C, B);
  C.print_formatted(std::cout);
}
