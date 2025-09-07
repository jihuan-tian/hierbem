/**
 * \file lapack-matrix-fill.cc
 * \brief Verify filling a LAPACKFullMatrixExt from a source matrix.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-30
 */

#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double>         src_data{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  LAPACKFullMatrixExt<double> M_src;
  LAPACKFullMatrixExt<double>::Reshape(3, 4, src_data, M_src);

  {
    LAPACKFullMatrixExt<double> M(3, 4);
    M.fill(M_src, 1, 2, 1, 1, 1., false);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(3, 4);
    M.fill(M_src, 1, 2, 1, 1, 2., true);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(2, 2);
    M.fill(M_src, 0, 0, 2, 2, 1., false);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(2, 2);
    M.fill(M_src, 0, 0, 2, 2, 1., true);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  return 0;
}
