/**
 * \file transpose.cc
 * \brief Test in-place transpose of a LAPACKFullMatrixExt.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-20
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> values{1., 2., 3., 4., 5., 6.};

  LAPACKFullMatrixExt<double> A;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, values, A);

  std::cout << "A=\n";
  A.print_formatted(std::cout, 2, false, 5, "0");

  A.transpose();

  std::cout << "A^T=\n";
  A.print_formatted(std::cout, 2, false, 5, "0");

  return 0;
}
