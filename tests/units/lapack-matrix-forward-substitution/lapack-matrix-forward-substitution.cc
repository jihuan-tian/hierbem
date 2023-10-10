/**
 * \file lapack-matrix-forward-substitution.cc
 * \brief Verify forward substitution of a unit lower triangle matrix.
 *
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2021-10-16
 */

#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> L_values{1, 2, 4, 7, 0, 1, 5, 8, 0, 0, 1, 9, 0, 0, 0, 1};
  LAPACKFullMatrixExt<double> L;
  LAPACKFullMatrixExt<double>::Reshape(4, 4, L_values, L);
  L.set_property(LAPACKSupport::lower_triangular);

  {
    Vector<double> b({3, 6, 9, 10});
    L.solve_by_forward_substitution(b);
    b.print(std::cout, 8);
  }

  {
    Vector<double> b({3, 6, 9, 10});
    Vector<double> x;
    L.solve_by_forward_substitution(x, b);
    x.print(std::cout, 8);
  }
}
