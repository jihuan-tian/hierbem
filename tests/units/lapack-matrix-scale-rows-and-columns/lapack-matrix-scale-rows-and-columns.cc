/**
 * \file scale-rows-and-columns.cc
 * \brief Test scaling rows and columns of a LAPACKFullMatrixExt, which is
 * actually left and right multiplication with a diagonal matrix.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-20
 */

#include <deal.II/lac/vector.h>

#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  /**
   * Diagonal matrix.
   */
  std::vector<double>    V1{1., 2., 3., 4., 5.};
  dealii::Vector<double> V2(V1.begin(), V1.end());

  /**
   * Matrix dimension.
   */
  const unsigned int dim = 5;

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.scale_rows(V1);
    std::cout << "V1*A=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.LAPACKFullMatrix<double>::scale_rows(V2);
    std::cout << "V2*A=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.scale_columns(V1);
    std::cout << "A*V1=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> A;
    LAPACKFullMatrixExt<double>::ConstantMatrix(dim, dim, 1.0, A);

    A.scale_columns(V2);
    std::cout << "A*V2=\n";
    A.print_formatted(std::cout, 2, false, 5, "0");
  }

  return 0;
}
