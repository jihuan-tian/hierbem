/**
 * \file lapack-matrix-agglomeration.cc
 * \brief Verify the agglomeration of four full submatrices into a larger full
 * matrix.
 * \ingroup linalg
 * \author Jihuan Tian
 * \date 2021-07-08
 */

#include <iostream>

#include "lapack_full_matrix_ext.h"

int
main()
{
  std::vector<double> M11_values{1, 2, 3, 4, 5, 6};
  std::vector<double> M12_values{7, 8, 9, 10};
  std::vector<double> M21_values{11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<double> M22_values{20, 21, 22, 23, 24, 25};

  LAPACKFullMatrixExt<double> M11, M12, M21, M22;
  LAPACKFullMatrixExt<double>::Reshape(2, 3, M11_values, M11);
  LAPACKFullMatrixExt<double>::Reshape(2, 2, M12_values, M12);
  LAPACKFullMatrixExt<double>::Reshape(3, 3, M21_values, M21);
  LAPACKFullMatrixExt<double>::Reshape(3, 2, M22_values, M22);

  LAPACKFullMatrixExt<double> M(M11, M12, M21, M22);

  /**
   * Output the matrices.
   */
  M11.print_formatted_to_mat(std::cout, "M11", 5, false, 10, "0");
  M12.print_formatted_to_mat(std::cout, "M12", 5, false, 10, "0");
  M21.print_formatted_to_mat(std::cout, "M21", 5, false, 10, "0");
  M22.print_formatted_to_mat(std::cout, "M22", 5, false, 10, "0");
  M.print_formatted_to_mat(std::cout, "M", 5, false, 10, "0");

  return 0;
}
