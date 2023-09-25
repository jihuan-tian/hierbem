/**
 * \file lapack-matrix-scale-matrix.cc
 * \brief Verify the scale of a whole full matrix.
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2021-10-05
 */

#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double>         A_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  LAPACKFullMatrixExt<double> A;
  LAPACKFullMatrixExt<double>::Reshape(3, 3, A_data, A);
  A.print_formatted_to_mat(std::cout, "A_before_scaling", 8, false, 12, "0");

  double b = 3.5;
  A *= b;
  A.print_formatted_to_mat(std::cout, "A_after_scaling", 8, false, 12, "0");

  return 0;
}
