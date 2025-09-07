/**
 * \file lapack-matrix-init-with-zeros.cc
 * \brief Verify if a LAPACKFullMatrixExt is initialized with zeros.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-09-28
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  LAPACKFullMatrixExt<double> A(5, 5);
  A.print_formatted(std::cout, 5, false, 10, "0");

  return 0;
}
