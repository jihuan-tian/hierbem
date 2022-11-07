/**
 * \file lapack-matrix-local-to-submatrix.cc
 * \brief Verify the restriction of a local full matrix to sub full matrix.
 * \ingroup linalg testers
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <deal.II/base/types.h>

#include <iostream>

#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"

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
   * Create a sub full matrix of \p M_b by specifying its block cluster as a
   * subset of the block cluster \f$\tau \times \sigma\f$ for \p M_b.
   */
  std::array<types::global_dof_index, 2> tau_subset{7, 11};
  std::array<types::global_dof_index, 2> sigma_subset{10, 13};
  LAPACKFullMatrixExt<double>          M_b_submatrix(tau_subset,
                                            sigma_subset,
                                            M_b,
                                            tau,
                                            sigma);
  M_b_submatrix.print_formatted_to_mat(std::cout, "M_b_submatrix");

  return 0;
}
