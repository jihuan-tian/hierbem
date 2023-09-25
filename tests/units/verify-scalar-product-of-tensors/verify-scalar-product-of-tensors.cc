/**
 * \file verify-scalar-product-of-tensors.cc
 * \brief Verify the scalar product of two rank-1 tensors.
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-09-18
 */

#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>

#include <iostream>

#include "generic_functors.h"

using namespace dealii;

int
main()
{
  Vector<double> v1({1.1, 2.5, 7.2});
  Vector<double> v2({-5.6, 3.8, 9.7});

  std::cout << "v1 * v2 = " << v1 * v2 << std::endl;

  /**
   * Convert the two vectors to tensors and perform the scalar product.
   */
  std::cout << "t1 * t2 = "
            << scalar_product(VectorToTensor<3, double, Vector<double>>(v1),
                              VectorToTensor<3, double, Vector<double>>(v2))
            << std::endl;

  return 0;
}
