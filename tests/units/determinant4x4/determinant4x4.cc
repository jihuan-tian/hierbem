// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "linear_algebra/linalg.h"

using namespace HierBEM::LinAlg;
using namespace Catch::Matchers;

TEST_CASE("Calculate determinant of 4x4 matrix", "[linalg]")
{
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

  auto   res      = determinant4x4(matrix);
  double expected = 10160.0;

  REQUIRE_THAT(res, WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
}
