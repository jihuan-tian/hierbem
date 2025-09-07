/**
 * \file lapack-matrix-global-to-rkmatrix.cc
 * \brief Verify the restriction of a global full matrix to a rank-k submatrix.
 * \ingroup rkmatrices test_cases
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  /**
   * Create a full matrix with data.
   */
  const unsigned int          n = 20;
  LAPACKFullMatrixExt<double> M(n, n);
  double                      counter = 1.0;
  for (auto it = M.begin(); it != M.end(); it++)
    {
      (*it) = counter;
      counter += 1.0;
    }
  M.print_formatted_to_mat(std::cout, "M");

  std::array<types::global_dof_index, 2> tau{7, 11};
  std::array<types::global_dof_index, 2> sigma{9, 13};

  RkMatrix<double> rkmat_no_trunc(tau, sigma, M);
  rkmat_no_trunc.print_formatted_to_mat(
    std::cout, "rkmat_no_trunc", 8, false, 16, "0");
  RkMatrix<double> rk1mat(tau, sigma, 1, M);
  rk1mat.print_formatted_to_mat(std::cout, "rk1mat", 8, false, 16, "0");

  return 0;
}
