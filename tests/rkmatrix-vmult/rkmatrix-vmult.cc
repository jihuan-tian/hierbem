/**
 * \file rkmatrix-vmult.cc
 * \brief Verify the multiplication of a rank-k matrix with a vector.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <iostream>

#include "debug_tools.h"
#include "rkmatrix.h"

int
main()
{
  {
    std::vector<double>         M_values{1, 2, 3, 4, 5, 6, 7, 8, 9};
    LAPACKFullMatrixExt<double> M;
    LAPACKFullMatrixExt<double>::Reshape(3, 3, M_values, M);
    RkMatrix<double> M_rk(2, M);

    Vector<double> x(3);
    Vector<double> y(3);

    x(0) = 7;
    x(1) = 9;
    x(2) = 10;

    y(0) = 12;
    y(1) = 15;
    y(2) = 7;

    M_rk.vmult(y, x, false);
    print_vector_to_mat(std::cout, "y", y);
  }

  {
    std::vector<double>         M_values{1, 2, 3, 4, 5, 6, 7, 8, 9};
    LAPACKFullMatrixExt<double> M;
    LAPACKFullMatrixExt<double>::Reshape(3, 3, M_values, M);
    RkMatrix<double> M_rk(2, M);

    Vector<double> x(3);
    Vector<double> y(3);

    x(0) = 7;
    x(1) = 9;
    x(2) = 10;

    y(0) = 12;
    y(1) = 15;
    y(2) = 7;

    M_rk.vmult(y, x, true);
    print_vector_to_mat(std::cout, "y", y);
  }

  {
    RkMatrix<double> M_rk(3, 3, 0);

    Vector<double> x(3);
    Vector<double> y(3);

    x(0) = 7;
    x(1) = 9;
    x(2) = 10;

    y(0) = 12;
    y(1) = 15;
    y(2) = 7;

    M_rk.vmult(y, x, false);
    print_vector_to_mat(std::cout, "y", y);
  }

  {
    RkMatrix<double> M_rk(3, 3, 0);

    Vector<double> x(3);
    Vector<double> y(3);

    x(0) = 7;
    x(1) = 9;
    x(2) = 10;

    y(0) = 12;
    y(1) = 15;
    y(2) = 7;

    M_rk.vmult(y, x, true);
    print_vector_to_mat(std::cout, "y", y);
  }

  return 0;
}
