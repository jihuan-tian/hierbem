/**
 * \file rank-k-decompose.cc
 * \brief Test rank-k decomposition of a LAPACKFullMatrixExt.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-06-20
 */

#include "lapack_full_matrix_ext.h"

int
main()
{
  LAPACKFullMatrixExt<double> M_orig;
  std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M_orig);

  std::cout << "M=\n";
  M_orig.print_formatted(std::cout, 10, true, 20, "0");

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 1;

    M.rank_k_decompose(k, A, B, true);

    std::cout << "* k=1, left associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 1;

    M.rank_k_decompose(k, A, B, false);

    std::cout << "* k=1, right associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 2;

    M.rank_k_decompose(k, A, B, true);

    std::cout << "* k=2, left associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 2;

    M.rank_k_decompose(k, A, B, false);

    std::cout << "* k=2, right associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 3;

    M.rank_k_decompose(k, A, B, true);

    std::cout << "* k=3, left associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 3;

    M.rank_k_decompose(k, A, B, false);

    std::cout << "* k=3, right associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 4;

    M.rank_k_decompose(k, A, B, true);

    std::cout << "* k=4, left associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  {
    LAPACKFullMatrixExt<double> M(M_orig);
    LAPACKFullMatrixExt<double> A, B;
    const unsigned int          k = 4;

    M.rank_k_decompose(k, A, B, false);

    std::cout << "* k=4, right associative\n";
    std::cout << "A=\n";
    A.print_formatted(std::cout, 10, true, 20, "0");
    std::cout << "B=\n";
    B.print_formatted(std::cout, 10, true, 20, "0");
  }

  return 0;
}
