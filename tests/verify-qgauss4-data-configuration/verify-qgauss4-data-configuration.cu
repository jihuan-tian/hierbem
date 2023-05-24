/**
 * @file verify-qgauss4-data-configuration.cc
 * @brief Verify the data configuration in @p QGauss<4>, which should be the
 * tensor product of two @p QGauss<2> objects with the iteration index for the
 * first one runs faster.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-03-22
 */

#include <iostream>

#include "debug_tools.hcu"
#include "quadrature.templates.h"

using namespace std;
using namespace dealii;
using namespace IdeoBEM;

int
main()
{
  const unsigned int dim        = 2;
  const unsigned int quad_order = 2;
  /**
   * @internal The argument in the constructor is the number of quadrature
   * points in each dimension, which should be @p quad_order+1.
   */
  QGauss<dim * 2> quad(quad_order + 1);
  print_qgauss(cout, quad, " ");

  QGauss<dim> quad1(quad_order + 1);
  QGauss<dim> quad2(quad_order + 1);

  /**
   * @internal Print out quadrature points and weights in 4D by taking the
   * tensor product of @p quad1 and @p quad2.
   */
  unsigned int counter = 0;
  for (unsigned int j = 0; j < quad2.size(); j++)
    {
      for (unsigned int i = 0; i < quad1.size(); i++)
        {
          cout << "#" << counter << ": points=(";
          cout << quad1.point(i) << " ";
          cout << quad2.point(j) << "), weights=";
          cout << quad1.weight(i) * quad2.weight(j) << endl;

          counter++;
        }
    }

  return 0;
}
