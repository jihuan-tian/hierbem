// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-qgauss4-data-configuration.cc
 * @brief Verify the data configuration in @p QGauss<4>, which should be the
 * tensor product of two @p QGauss<2> objects with the iteration index for the
 * first one runs faster.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-03-22
 */

#include <iostream>

#include "quadrature/quadrature.templates.h"
#include "utilities/debug_tools.h"

using namespace std;
using namespace dealii;
using namespace HierBEM;

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
