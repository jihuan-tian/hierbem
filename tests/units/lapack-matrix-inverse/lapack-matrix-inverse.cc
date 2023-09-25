/**
 * \file lapack-matrix-inverse.cc
 * \brief Verify matrix inverse computed by Gauss elimination, which is compared
 * with the standard function in deal.ii.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-08-30
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> M_data{-1, 5,  2,  -3, 6,  1,  -2, 4,  2,  -3, -4, 1,  -3,
                             -1, 1,  2,  -2, 4,  2,  -1, 3,  1,  -1, 3,  -3, 7,
                             2,  -3, 7,  2,  -2, 2,  1,  0,  0,  -1, 1,  -4, 0,
                             0,  0,  2,  0,  -2, 3,  -1, -1, 6,  -2, 4,  3,  -2,
                             4,  -1, -1, 3,  3,  -4, -6, 1,  -3, -3, 1,  -2};
  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(8, 8, M_data, M);
  M.print_formatted_to_mat(std::cout, "M", 8, false, 16, "0");

  /**
   * Calculate the inverse form an input matrix.
   */
  LAPACKFullMatrixExt<double> M_inv_from_input;
  M_inv_from_input.invert(M);
  M_inv_from_input.print_formatted_to_mat(
    std::cout, "M_inv_from_input", 12, false, 20, "0");

  /**
   * Make a copy of the matrix for manual calculation of its inverse.
   */
  LAPACKFullMatrixExt<double> M_prime(M);

  /**
   * Calculate the matrix inverse using deal.ii \p invert function.
   */
  M.LAPACKFullMatrix<double>::invert();
  M.print_formatted_to_mat(std::cout, "M_inv", 12, false, 20, "0");

  /**
   * Calculate the matrix inverse using Gauss elimination.
   */
  LAPACKFullMatrixExt<double> M_prime_inv;
  M_prime.invert_by_gauss_elim(M_prime_inv);
  M_prime_inv.print_formatted_to_mat(
    std::cout, "M_prime_inv", 16, false, 25, "0");

  return 0;
}
