/**
 * \file aca-random-select-ref-row.cc
 * \brief Verify random reference row selection used in ACA+.
 * \ingroup testers aca
 * \author Jihuan Tian
 * \date 2022-03-09
 */

#include <forward_list>
#include <iostream>
#include <iterator>
#include <vector>

#include "hmatrix/aca_plus/aca_plus.hcu"

using namespace HierBEM;
using namespace dealii;

int
main()
{
  std::vector<double> a(64);
  gen_linear_indices<vector_uta, double>(a, 1.3, 2.4);
  LAPACKFullMatrixExt<double> A;
  LAPACKFullMatrixExt<double>::Reshape(8, 8, a, A);

  std::forward_list<unsigned int> remaining_row_indices{0, 1, 2, 3, 4, 5, 6, 7};

  unsigned int   ref_row_index = 1;
  Vector<double> row_vector(8);
  for (unsigned int i = 0; i < 8; i++)
    {
      ref_row_index = random_select_ref_row(row_vector,
                                            A,
                                            remaining_row_indices,
                                            ref_row_index);
      std::cout << "Current ref row: " << ref_row_index << std::endl;
      std::cout << "Current row vector: ";
      row_vector.print(std::cout, 10, false);
    }
  std::cout << "Final size of remaining_row_indices: "
            << size(remaining_row_indices) << std::endl;

  return 0;
}
