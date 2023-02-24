/**
 * @file cudafullmatrix-det2x2.cu
 * @brief Verify the calculation of the determinant of a 2x2 matrix.
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

  CUDAFullMatrix<double> A(data, 2, 2);
  cout << "det(A)=" << A.determinant2x2() << endl;

  return 0;
}
