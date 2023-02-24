/**
 * @file cudafullmatrix-Tvmult.cu
 * @brief Verify transposed matrix-vector multiplication.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <iostream>

#include "cu_fullmatrix.hcu"
#include "cu_vector.hcu"

using namespace std;
using namespace IdeoBEM::CUDAWrappers;

int
main()
{
  double A_data[12]{2, 8, 9, 7, 1, 3, 11, 20, 13, 20, 30, 10};
  double v_data[3]{7, 3, 10};
  double w_data[4];
  double w_adding_data[4]{1, 2, 3, 20};

  CUDAFullMatrix<double> A(A_data, 3, 4);
  CUDAVector<double>     v(v_data, 3);
  CUDAVector<double>     w(w_data, 4);
  CUDAVector<double>     w_adding(w_adding_data, 4);

  A.Tvmult(w, v);
  A.Tvmult(w_adding, v, true);

  cout << "w=\n";
  w.print(false);

  cout << "w_adding=\n";
  w_adding.print(false);

  return 0;
}
