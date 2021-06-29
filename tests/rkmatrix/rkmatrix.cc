/**
 * \file rkmatrix.cc
 * \brief Test RkMatrix class.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-20
 */

#include "rkmatrix.h"

#include <iostream>

int
main()
{
  LAPACKFullMatrixExt<double> M;
  std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M);
  std::cout << "M=\n";
  M.print_formatted(std::cout, 5, false, 10, "0");

  std::vector<unsigned int> tau{0, 2};
  std::vector<unsigned int> sigma{1, 2, 3};

  {
    RkMatrix<double> A(tau, sigma, 1, M);
    std::cout << "Rank-1 matrix:\n";
    A.print_formatted(std::cout, 5, false, 10, "0");

    LAPACKFullMatrixExt<double> A_full;
    A.convertToFullMatrix(A_full);
    std::cout << "A converted back to full matrix:\n";
    A_full.print_formatted(std::cout, 5, false, 10, "0");
  }

  {
    RkMatrix<double> A(tau, sigma, 2, M);
    std::cout << "Rank-2 matrix:\n";
    A.print_formatted(std::cout, 5, false, 10, "0");

    LAPACKFullMatrixExt<double> A_full;
    A.convertToFullMatrix(A_full);
    std::cout << "A converted back to full matrix:\n";
    A_full.print_formatted(std::cout, 5, false, 10, "0");

    /**
     * Finally, let's write the RkMatrix in the Octave text data format.
     */
    print_rkmatrix_to_mat(std::cout, "A", A);
  }

  return 0;
}
