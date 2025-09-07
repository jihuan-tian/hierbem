/**
 * \file lapack-matrix-read-from-mat.cc
 * \brief Verify reading a matrix from a file saved from Octave in text format,
 * i.e. saved with the option \p -text.
 *
 * \ingroup test_cases linalg
 * \author Jihuan Tian
 * \date 2021-10-20
 */

#include <fstream>

#include "linear_algebra/lapack_full_matrix_ext.h"

int
main()
{
  std::ifstream input("input.mat");

  LAPACKFullMatrixExt<double> M;
  M.read_from_mat(input, "M");
  M.print_formatted_to_mat(std::cout, "M_read", 15, false, 25, "0");

  return 0;
}
