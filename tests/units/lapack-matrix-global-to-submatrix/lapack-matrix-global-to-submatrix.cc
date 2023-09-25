/**
 * \file lapack-matrix-global-to-submatrix.cc
 * \brief Verify the restriction of a global full matrix to sub full matrix.
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2021-07-28
 */

#include <deal.II/base/types.h>

#include <iostream>

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

  std::array<types::global_dof_index, 2> tau{7,11};
  std::array<types::global_dof_index, 2> sigma{3,7};
  LAPACKFullMatrixExt<double>          M_b(tau, sigma, M);

  M_b.print_formatted_to_mat(std::cout, "M_b");

  return 0;
}
