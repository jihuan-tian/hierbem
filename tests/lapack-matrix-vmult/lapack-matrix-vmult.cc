/**
 * \file lapack-matrix-vmult.cc
 * \brief Verify matrix-vector multiplication.
 *
 * \ingroup testers linalg
 * \author Jihuan Tian
 * \date 2022-05-13
 */

#include <deal.II/lac/vector.h>

#include <iostream>
#include <vector>

#include "debug_tools.h"
#include "generic_functors.h"
#include "lapack_full_matrix_ext.h"
#include "unary_template_arg_containers.h"

using namespace dealii;

int
main()
{
  std::vector<double> v(36);
  gen_linear_indices<vector_uta, double>(v, 1.1, 0.45);

  LAPACKFullMatrixExt<double> M;
  LAPACKFullMatrixExt<double>::Reshape(6, 6, v, M);
  M.print_formatted_to_mat(std::cout, "M", 15, false, 25, "0");

  Vector<double> x({7, 3, 4, 10, 22, 15});
  Vector<double> y(6);
  Vector<double> z(6);

  /**
   * Perform a normal matrix-vector multiplication.
   */
  M.vmult(y, x);

  /**
   * Perform a symmetric matrix-vector multiplication, where only the lower
   * triangular part of the matrix is used.
   */
  M.set_property(LAPACKSupport::symmetric);
  M.vmult(z, x);

  /**
   * Output vectors.
   */
  print_vector_to_mat(std::cout, "x", x);
  print_vector_to_mat(std::cout, "y", y);
  print_vector_to_mat(std::cout, "z", z);

  return 0;
}
