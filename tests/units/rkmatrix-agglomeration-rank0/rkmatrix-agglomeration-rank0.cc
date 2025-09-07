/**
 * \file rkmatrix-agglomeration-rank0.cc
 * \brief Verify the agglomeration of four rank-k submatrices into a larger
 * rank-k matrix. Some of the submatrices have rank zero.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2021-10-09
 */

#include <iostream>

#include "hmatrix/rkmatrix.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  // std::vector<double> M11_values{1, 2, 3, 4, 5, 6};
  std::vector<double> M11_values{0, 0, 0, 0, 0, 0};
  std::vector<double> M12_values{7, 8, 9, 10};
  std::vector<double> M21_values{11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<double> M22_values{20, 21, 22, 23, 24, 25};

  LAPACKFullMatrixExt<double> M11, M12, M21, M22;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, M11_values, M11);
  LAPACKFullMatrixExt<double>::Reshape(2, 2, M12_values, M12);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, M21_values, M21);
  LAPACKFullMatrixExt<double>::Reshape(3, 2, M22_values, M22);

  LAPACKFullMatrixExt<double> M(M11, M12, M21, M22);

  M11.print_formatted_to_mat(std::cout, "M11", 15, false, 25, "0");
  M12.print_formatted_to_mat(std::cout, "M12", 15, false, 25, "0");
  M21.print_formatted_to_mat(std::cout, "M21", 15, false, 25, "0");
  M22.print_formatted_to_mat(std::cout, "M22", 15, false, 25, "0");
  M.print_formatted_to_mat(std::cout, "M", 15, false, 25, "0");

  const unsigned int fixed_rank_k = 4;

  RkMatrix<double> M11_rk(fixed_rank_k, M11);
  RkMatrix<double> M12_rk(fixed_rank_k, M12);
  RkMatrix<double> M21_rk(fixed_rank_k, M21);
  RkMatrix<double> M22_rk(fixed_rank_k, M22);

  RkMatrix<double> M_rk(fixed_rank_k, M11_rk, M12_rk, M21_rk, M22_rk, 1.5);

  /**
   * Output the matrices.
   */
  // M11_rk.print_formatted_to_mat(std::cout, "M11_rk", 15, false, 25, "0");
  M12_rk.print_formatted_to_mat(std::cout, "M12_rk", 15, false, 25, "0");
  M21_rk.print_formatted_to_mat(std::cout, "M21_rk", 15, false, 25, "0");
  M22_rk.print_formatted_to_mat(std::cout, "M22_rk", 15, false, 25, "0");
  M_rk.print_formatted_to_mat(std::cout, "M_rk", 15, false, 25, "0");

  return 0;
}
