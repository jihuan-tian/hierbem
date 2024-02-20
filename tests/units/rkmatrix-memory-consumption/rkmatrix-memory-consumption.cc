/**
 * \file rkmatrix-memory-consumption.cc
 * \brief Verify the memory consumption calculation for a rank-k matrix.
 *
 * \ingroup testers rkmatrices
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <iostream>

#include "rkmatrix.h"

using namespace HierBEM;

int
main()
{
  LAPACKFullMatrixExt<double> M;
  std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  LAPACKFullMatrixExt<double>::Reshape(3, 5, values, M);
  std::cout << "M=\n";
  M.print_formatted(std::cout, 5, false, 10, "0");

  std::array<types::global_dof_index, 2> tau{{0, 2}};
  std::array<types::global_dof_index, 2> sigma{{1, 4}};

  RkMatrix<double> A(tau, sigma, 2, M);
  std::cout << "Rank-2 matrix:\n";
  A.print_formatted(std::cout, 5, false, 10, "0");

  std::cout << "Memory consumption of A.A: " << A.get_A().memory_consumption()
            << "\n";
  std::cout << "Memory consumption of A.B: " << A.get_B().memory_consumption()
            << "\n";
  std::cout << "Memory consumption of rank-k matrix A: "
            << A.memory_consumption() << std::endl;

  return 0;
}
