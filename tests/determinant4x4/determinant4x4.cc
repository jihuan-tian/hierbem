#include <deal.II/base/logstream.h>

#include "linalg.h"

using namespace LinAlg;
using namespace dealii;

int
main()
{
  deallog.depth_console(2);
  deallog.pop();

  FullMatrix<double> matrix(4, 4);
  matrix(0, 0) = 3.;
  matrix(0, 1) = 10.;
  matrix(0, 2) = 5.;
  matrix(0, 3) = 16.;
  matrix(1, 0) = 2.;
  matrix(1, 1) = 10.;
  matrix(1, 2) = 8.;
  matrix(1, 3) = 35.;
  matrix(2, 0) = 17.;
  matrix(2, 1) = 66.;
  matrix(2, 2) = 19.;
  matrix(2, 3) = 20.;
  matrix(3, 0) = 9.;
  matrix(3, 1) = 20.;
  matrix(3, 2) = 13.;
  matrix(3, 3) = 4.;

  deallog << "Matrix determinant is " << determinant4x4(matrix) << std::endl;

  return 0;
}
