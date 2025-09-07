/**
 * \file rkmatrix-to-fullmatrix-rank0.cc
 * \brief Verify the conversion of a rank-0 matrix to a full matrix.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"

int
main()
{
  RkMatrix<double>            M_rk(3, 5, 0);
  LAPACKFullMatrixExt<double> M;
  M_rk.convertToFullMatrix(M);

  M.print_formatted_to_mat(std::cout, "M", 5, false, 8, "0");

  return 0;
}
