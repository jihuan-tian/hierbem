// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file full-matrix-wrapper-det3x3.cu
 * @brief Verify the calculation of the determinant of a 3x3 matrix.
 *
 * @ingroup test_cases linalg
 * @author Jihuan Tian
 * @date 2023-02-24
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "platform_shared/full_matrix_wrapper.h"

using namespace HierBEM;
using namespace Catch::Matchers;

TEST_CASE("Calculate determinant of 3x3 matrix by CUDA",
          "[full_matrix_wrapper]")
{
  double                                    data[9]{std::tan(1),
                 std::tan(2),
                 std::tan(3),
                 std::tan(4),
                 std::tan(5),
                 std::tan(6),
                 std::tan(7),
                 std::tan(8),
                 std::tan(9)};
  PlatformShared::FullMatrixWrapper<double> A(data, 3, 3);

  auto   res      = A.determinant3x3();
  double expected = -0.588238;

  REQUIRE_THAT(res, WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
}
