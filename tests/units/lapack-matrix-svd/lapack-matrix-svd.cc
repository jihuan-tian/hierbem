/**
 * \file svd.cc
 * \brief Test singular value decomposition (SVD) and reduced SVD.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-19
 */

#include "debug_tools.h"
#include "lapack_full_matrix_ext.h"

using namespace dealii;

int
main()
{
  const unsigned int m = 3;
  const unsigned int n = 5;

  LAPACKFullMatrixExt<double> A_original(m, n);
  LAPACKFullMatrixExt<double> U, VT;
  std::vector<double>         Sigma_r;

  unsigned int counter = 1;
  for (unsigned int i = 0; i < m; i++)
    {
      for (unsigned int j = 0; j < n; j++)
        {
          A_original(i, j) = (double)counter;
          counter++;
        }
    }

  A_original.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "Original SVD without rank truncation\n";
    A.svd(U, Sigma_r, VT);

    U.print_formatted_to_mat(std::cout, "U1", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT1", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r1", Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 1\n";

    A.svd(U, Sigma_r, VT, 1);

    U.print_formatted_to_mat(std::cout, "U2", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT2", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r2", Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 2\n";

    A.svd(U, Sigma_r, VT, 2);

    U.print_formatted_to_mat(std::cout, "U3", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT3", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r3", Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 3\n";

    A.svd(U, Sigma_r, VT, 3);

    U.print_formatted_to_mat(std::cout, "U4", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT4", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r4", Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 4\n";

    A.svd(U, Sigma_r, VT, 4);

    U.print_formatted_to_mat(std::cout, "U5", 8, false, 16, "0");
    VT.print_formatted_to_mat(std::cout, "VT5", 8, false, 16, "0");
    print_vector_to_mat(std::cout, "Sigma_r5", Sigma_r);
  }

  return 0;
}
