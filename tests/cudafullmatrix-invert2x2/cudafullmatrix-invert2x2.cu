/**
 * @file cudafullmatrix-invert2x2.cu
 * @brief Verify the calculation of the invert of a 2x2 matrix.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "cu_fullmatrix.hcu"

using namespace IdeoBEM::CUDAWrappers;
using namespace std;

int
main()
{
  double data[4]{1, 2, 3, 4};
  double data_inv[4];

  CUDAFullMatrix<double> A(data, 2, 2);
  CUDAFullMatrix<double> A_inv(data_inv, 2, 2);

  A_inv.invert2x2(A);

  A_inv.print();

  return 0;
}
