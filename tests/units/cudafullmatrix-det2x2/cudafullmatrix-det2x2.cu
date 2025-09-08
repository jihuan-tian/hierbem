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
 * @file cudafullmatrix-det2x2.cu
 * @brief Verify the calculation of the determinant of a 2x2 matrix.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-24
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <iostream>

#include "linear_algebra/cu_fullmatrix.hcu"

using namespace HierBEM::CUDAWrappers;
using namespace Catch::Matchers;

TEST_CASE("Calculate determinant of 2x2 matrix by CUDA", "[cuda_full_matrix]")
{
  double                 data[4]{1, 2, 3, 4};
  CUDAFullMatrix<double> A(data, 2, 2);

  auto   res      = A.determinant2x2();
  double expected = -2.;

  REQUIRE_THAT(res, WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
}
