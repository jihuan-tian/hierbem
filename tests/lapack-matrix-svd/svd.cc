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

  std::cout << "Before SVD: A=\n";
  A_original.print_formatted(std::cout);

  {
    LAPACKFullMatrixExt<double> A(A_original);

    A.svd(U, Sigma_r, VT);

    std::cout << "After SVD: A=\n";
    A.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 1\n";

    A.svd(U, Sigma_r, VT, 1);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 2\n";

    A.svd(U, Sigma_r, VT, 2);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 3\n";

    A.svd(U, Sigma_r, VT, 3);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "SVD with rank truncated to 4\n";

    A.svd(U, Sigma_r, VT, 4);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 1\n";

    A.reduced_svd(U, Sigma_r, VT, 1);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 2\n";

    A.reduced_svd(U, Sigma_r, VT, 2);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 3\n";

    A.reduced_svd(U, Sigma_r, VT, 3);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  {
    LAPACKFullMatrixExt<double> A(A_original);

    std::cout << "RSVD with rank truncated to 4\n";

    A.reduced_svd(U, Sigma_r, VT, 4);

    std::cout << "U=\n";
    U.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "VT=\n";
    VT.print_formatted(std::cout, 5, false, 10, "0");

    std::cout << "Sigma_r=\n";
    print_vector_values(std::cout, Sigma_r);
  }

  return 0;
}
