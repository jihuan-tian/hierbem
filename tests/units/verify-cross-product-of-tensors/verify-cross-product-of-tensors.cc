// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file verify-cross-product-of-tensors.cc
 * \brief Verify the cross product in 3D of two rank-1 tensors.
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-09-18
 */

#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>

#include <iostream>

#include "utilities/generic_functors.h"

using namespace dealii;

int
main()
{
  Tensor<1, 3, double> t1({1.1, 2.5, 7.2});
  Tensor<1, 3, double> t2({-5.6, 3.8, 9.7});

  std::cout << "t1 x t2 = " << cross_product_3d(t1, t2) << std::endl;

  return 0;
}
