/**
 * \file lapack-matrix-fill-rows.cc
 * \brief Verify filling the rows of a @p LAPACKFullMatrixExt.
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2022-11-02
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
    LAPACKFullMatrixExt<double> M(10, 4);
    M.fill_rows({1, 11}, M_src, {5, 8}, 2.0, false);
    std::cout << "M=\n";
    M.print_formatted(std::cout, 2, false, 5, "0");
  }

  return 0;
}
