/**
 * \file verify-table-copy-constructor.cc
 * \brief Verify the copy constructor (deep copy) of @p Table<N,T>.
 *
 * \ingroup test_cases dealii_verify
 * \author Jihuan Tian
 * \date 2022-07-14
 */

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>

#include <deal.II/lac/full_matrix.h>

#include <iostream>

#include "utilities/debug_tools.h"

using namespace dealii;

int
main()
{
  Table<2, FullMatrix<double>> A;
  A.reinit(TableIndices<2>(2, 2));
  A(TableIndices<2>(1, 1)).reinit(10, 10);

  for (unsigned int i = 0; i < 10; i++)
    {
      for (unsigned int j = 0; j < 10; j++)
        {
          A(TableIndices<2>(1, 1))(i, j) = i + j;
        }
    }

  Table<2, FullMatrix<double>> B(A);
  print_matrix_to_mat(
    std::cout, "M", B(TableIndices<2>(1, 1)), 10, false, 15, "0");
}
