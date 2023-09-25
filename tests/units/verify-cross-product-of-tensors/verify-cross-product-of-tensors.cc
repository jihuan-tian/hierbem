/**
 * \file verify-cross-product-of-tensors.cc
 * \brief Verify the cross product in 3D of two rank-1 tensors.
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
  Tensor<1, 3, double> t1({1.1, 2.5, 7.2});
  Tensor<1, 3, double> t2({-5.6, 3.8, 9.7});

  std::cout << "t1 x t2 = " << cross_product_3d(t1, t2) << std::endl;

  return 0;
}
