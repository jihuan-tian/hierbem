/**
 * \file rkmatrix-local-to-rkmatrix.cc
 * \brief Verify the restriction of a local rank-k matrix to a rank-k submatrix.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include "hmatrix/rkmatrix.h"

#include <iostream>

#include "hmatrix/hmatrix.h"

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
   * Create a rank-k matrix by restriction from the global full matrix on the
   * block cluster \f$\tau \times \sigma\f$.
   */
  std::vector<types::global_dof_index> tau{2, 3, 4, 5, 7, 10, 18, 19};
  std::vector<types::global_dof_index> sigma{3, 4, 8, 9, 11, 13, 15, 16, 17};
  RkMatrix<double>                     M_b_rk(tau, sigma, M);
  M_b_rk.print_formatted_to_mat(std::cout, "M_b_rk", 8, false, 16, "0");

  /**
   * Build the maps from global row and column indices respectively to local
   * indices wrt. M_b.
   */
  std::map<types::global_dof_index, size_t> row_index_global_to_local_map;
  std::map<types::global_dof_index, size_t> col_index_global_to_local_map;
  build_index_set_global_to_local_map(tau, row_index_global_to_local_map);
  build_index_set_global_to_local_map(sigma, col_index_global_to_local_map);

  /**
   * Create a rank-k submatrix of \p M_b by specifying its block cluster as a
   * subset of the block cluster \f$\tau \times \sigma\f$ for \p M_b.
   */
  std::vector<types::global_dof_index> tau_subset{3, 7, 10, 19};
  std::vector<types::global_dof_index> sigma_subset{8, 13, 17};
  RkMatrix<double>                     rkmat_no_trunc(tau_subset,
                                  sigma_subset,
                                  M_b_rk,
                                  row_index_global_to_local_map,
                                  col_index_global_to_local_map);
  rkmat_no_trunc.print_formatted_to_mat(
    std::cout, "rkmat_no_trunc", 8, false, 16, "0");

  RkMatrix<double> rk1mat(tau_subset,
                          sigma_subset,
                          1,
                          M_b_rk,
                          row_index_global_to_local_map,
                          col_index_global_to_local_map);
  rk1mat.print_formatted_to_mat(std::cout, "rk1mat", 8, false, 16, "0");

  return 0;
}
