/**
 * @file lapack-matrix-keep-first-n-columns.cc
 * @brief
 *
 * @ingroup linalg testers
 * @author
 * @date 2024-01-31
 */

#include <iostream>

#include "generic_functors.h"
#include "lapack_full_matrix_ext.h"
#include "unary_template_arg_containers.h"

using namespace std;
using namespace HierBEM;

int
main()
{
  std::vector<double> v(35);
  gen_linear_indices<vector_uta, double>(v, 1, 0.5);
  LAPACKFullMatrixExt<double> A, B;

  for (int i = 5; i >= 0; i--)
    {
      LAPACKFullMatrixExt<double>::Reshape(7, 5, v, A);
      A.keep_first_n_columns(i, false);
      A.print_formatted_to_mat(std::cout, "A", 8, false, 16, "0");
    }

  for (int i = 5; i >= 0; i--)
    {
      LAPACKFullMatrixExt<double>::Reshape(7, 5, v, B);
      B.keep_first_n_columns(i, true);
      B.print_formatted_to_mat(std::cout, "B", 8, false, 16, "0");
    }

  return 0;
}
