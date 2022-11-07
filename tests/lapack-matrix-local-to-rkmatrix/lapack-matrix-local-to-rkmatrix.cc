/**
 * \file lapack-matrix-local-to-rkmatrix.cc
 * \brief Verify the restriction of a local full matrix to a rank-k submatrix.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <iostream>

#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "rkmatrix.h"

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
   * Create a local matrix as a sub matrix of the original matrix.
   */
  std::array<types::global_dof_index, 2> tau{5, 13};
  std::array<types::global_dof_index, 2> sigma{7, 15};
  LAPACKFullMatrixExt<double>          M_b(tau, sigma, M);
  M_b.print_formatted_to_mat(std::cout, "M_b");

  /**
   * Create a sub rank-k matrix of \p M_b by specifying its block cluster as a
   * subset of the block cluster \f$\tau \times \sigma\f$ for \p M_b.
   */
  std::array<types::global_dof_index, 2> tau_subset{7, 11};
  std::array<types::global_dof_index, 2> sigma_subset{10, 13};
  RkMatrix<double>                     rkmat_no_trunc(tau_subset,
                                                      sigma_subset,
                                                      M_b,
                                                      tau,
                                                      sigma);
  rkmat_no_trunc.print_formatted_to_mat(
    std::cout, "rkmat_no_trunc", 8, false, 16, "0");

  RkMatrix<double> rk1mat(tau_subset,
                          sigma_subset,
                          1,
                          M_b,
                          tau,
                          sigma);
  rk1mat.print_formatted_to_mat(std::cout, "rk1mat", 8, false, 16, "0");

  return 0;
}
