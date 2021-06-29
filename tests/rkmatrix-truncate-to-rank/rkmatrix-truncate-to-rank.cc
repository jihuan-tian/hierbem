/**
 * \file rkmatrix-truncate-to-rank.cc
 * \brief Verify the truncation of an RkMatrix to a given rank.
 * \ingroup
 * \author Jihuan Tian
 * \date 2021-06-24
 */

#include "debug_tools.h"
#include "rkmatrix.h"

int
main()
{
  LAPACKFullMatrixExt<double> M;
  std::vector<double>         values{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                             11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  LAPACKFullMatrixExt<double>::Reshape(4, 5, values, M);
  print_matrix_to_mat(std::cout, "M", M);

  RkMatrix<double> A(3, M);
  print_rkmatrix_to_mat(std::cout, "A", A);

  {
    RkMatrix<double> A_trunc(A);
    A_trunc.truncate_to_rank(3);
    print_rkmatrix_to_mat(std::cout, "A_trunc_to_3", A_trunc);
  }

  {
    RkMatrix<double> A_trunc(A);
    A_trunc.truncate_to_rank(2);
    print_rkmatrix_to_mat(std::cout, "A_trunc_to_2", A_trunc);
  }

  {
    RkMatrix<double> A_trunc(A);
    A_trunc.truncate_to_rank(1);
    print_rkmatrix_to_mat(std::cout, "A_trunc_to_1", A_trunc);
  }

  return 0;
}
