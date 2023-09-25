/**
 * @file cudafullmatrix-det3x3.cu
 * @brief Verify the calculation of the determinant of a 3x3 matrix.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <cmath>
#include <iostream>

#include "cu_fullmatrix.hcu"

using namespace HierBEM::CUDAWrappers;
using namespace std;

int
main()
{
  double data[9]{
    tan(1), tan(2), tan(3), tan(4), tan(5), tan(6), tan(7), tan(8), tan(9)};

  CUDAFullMatrix<double> A(data, 3, 3);
  cout << "det(A)=" << A.determinant3x3() << endl;

  return 0;
}
