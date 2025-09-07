/**
 * @file cudafullmatrix-mTmult.cu
 * @brief Verify matrix-matrix multiplication with the second operand
 * transposed.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "linear_algebra/cu_fullmatrix.hcu"

using namespace std;
using namespace HierBEM::CUDAWrappers;

int
main()
{
  double A_data[9]{2, 8, 9, 7, 1, 3, 11, 20, 13};
  double B_data[6]{1, 3, 5, 7, 9, 10};
  double C_data[6];
  double C_adding_data[6]{1, 1, 1, 2, 2, 2};

  CUDAFullMatrix<double> A(A_data, 3, 3);
  CUDAFullMatrix<double> B(B_data, 2, 3);
  CUDAFullMatrix<double> C(C_data, 3, 2);
  CUDAFullMatrix<double> C_adding(C_adding_data, 3, 2);

  A.mTmult(C, B);
  A.mTmult(C_adding, B, true);

  cout << "C=\n";
  C.print(false, false);

  cout << "C_adding=\n";
  C_adding.print(false, false);

  return 0;
}
