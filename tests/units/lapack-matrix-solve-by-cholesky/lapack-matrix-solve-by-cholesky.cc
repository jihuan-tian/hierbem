/**
 * \file lapack-matrix-solve-by-cholesky.cc
 * \brief Verify solving a full matrix using Cholesky decomposition.
 *
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2021-10-22
 */

#include <deal.II/lac/vector.h>

#include <fstream>
#include <iostream>

#include "debug_tools.h"
#include "lapack_full_matrix_ext.h"
#include "read_octave_data.h"

int
main()
{
  LAPACKFullMatrixExt<double> M;
  std::ifstream               in("M.dat");
  M.read_from_mat(in, "M");


  dealii::Vector<double> b({1, 3, 2, 4, 5, 7, 9, 6});
  print_vector_to_mat(std::cout, "b", b, false);

  /**
   * Compute Cholesky decomposition of the matrix.
   */
  M.set_property(LAPACKSupport::symmetric);
  M.compute_cholesky_factorization();

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
