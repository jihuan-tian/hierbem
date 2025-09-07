/**
 * \file lapack-matrix-add-without-factor.cc
 * \brief Verify matrix addition \f$C = A + B\f$
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-05-03
 */

#include <cmath>
#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

using namespace HierBEM;

int
main()
{
  std::vector<double> A_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B_data{3, 5, 7, 4, 6, 8, 5, 7, 9};

  LAPACKFullMatrixExt<double> A, B;
  LAPACKFullMatrixExt<double>::Reshape(3, 3, A_data, A);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, B_data, B);

  A.print_formatted_to_mat(std::cout, "A", 8, false, 12, "0");
  B.print_formatted_to_mat(std::cout, "B", 8, false, 12, "0");

  /**
   * Add matrix @p B into @p A.
   */
  A.add(B);
  A.print_formatted_to_mat(std::cout, "A_self_added", 8, false, 12, "0");

  return 0;
}
