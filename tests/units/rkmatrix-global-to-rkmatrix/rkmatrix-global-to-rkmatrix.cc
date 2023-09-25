/**
 * \file rkmatrix-global-to-rkmatrix.cc
 * \brief Verify the restriction of a global rank-k matrix to a rank-k
 * submatrix.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include "rkmatrix.h"

#include <iostream>

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

  /**
   * Create a rank-k matrix from the full matrix.
   */
  RkMatrix<double> M_rk(2, M);
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 8, false, 16, "0");

  /**
   * Create a rank-k matrix by restriction from the large rank-k matrix on the
   * block cluster \f$\tau \times \sigma\f$.
   */
  std::vector<types::global_dof_index> tau{2, 3, 4, 5, 7, 10, 18, 19};
  std::vector<types::global_dof_index> sigma{3, 4, 8, 9, 11, 13, 15, 16, 17};
  RkMatrix<double>                     M_b_rk(tau, sigma, M_rk);
  M_b_rk.print_formatted_to_mat(std::cout, "M_b_rk", 8, false, 16, "0");

  RkMatrix<double> M_b_rk1(tau, sigma, 1, M_rk);
  M_b_rk1.print_formatted_to_mat(std::cout, "M_b_rk1", 8, false, 16, "0");

  return 0;
}
