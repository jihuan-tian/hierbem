/**
 * \file lapack-matrix-backward-substitution.cc
 * \brief Verify backward substitution of an upper triangle matrix.
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-16
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> L_values{1, 2, 4, 7, 0, 3, 5, 8, 0, 0, 6, 9, 0, 0, 0, 10};
  /**
   * The transposition of the matrix is to be solved.
   */
  LAPACKFullMatrixExt<double> L;
  LAPACKFullMatrixExt<double>::Reshape(4, 4, L_values, L);
  L.set_property(LAPACKSupport::lower_triangular);

  {
    Vector<double> b({3, 6, 9, 10});
    L.solve_by_backward_substitution(b, true);
    b.print(std::cout, 8);
  }

  {
    Vector<double> b({3, 6, 9, 10});
    Vector<double> x;
    L.solve_by_backward_substitution(x, b, true);
    x.print(std::cout, 8);
  }
}
