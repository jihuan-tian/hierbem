/**
 * \file lapack-matrix-solve-by-lu.cc
 * \brief Verify solving a full matrix using LU decomposition.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-21
 */

#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

#include "linear_algebra/lapack_full_matrix_ext.h"
#include "utilities/debug_tools.h"
#include "utilities/read_octave_data.h"

int
main()
{
  std::vector<double> M_data{-1, 5,  2,  -3, 6,  1,  -2, 4,  2,  -3, -4, 1,  -3,
                             -1, 1,  2,  -2, 4,  2,  -1, 3,  1,  -1, 3,  -3, 7,
                             2,  -3, 7,  2,  -2, 2,  1,  0,  0,  -1, 1,  -4, 0,
                             0,  0,  2,  0,  -2, 3,  -1, -1, 6,  -2, 4,  3,  -2,
                             4,  -1, -1, 3,  3,  -4, -6, 1,  -3, -3, 1,  -2};
  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(8, 8, M_data, M);
  M.print_formatted_to_mat(std::cout, "M", 8, false, 16, "0");

  dealii::Vector<double> b({1, 3, 2, 4, 5, 7, 9, 6});
  print_vector_to_mat(std::cout, "b", b, false);

  /**
   * Compute LU decomposition of the matrix.
   */
  M.compute_lu_factorization();

  /**
   * Solve the matrix.
   */
  M.solve(b);

  /**
   * Print the result vector, which is stored in \p b.
   */
  print_vector_to_mat(std::cout, "x", b, false);

  return 0;
}
