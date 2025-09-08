// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-dealii-copy-point-double-to-point-float.cc
 * @brief Verify copying Point<spacedim, double> to Point<spacedim, float>.
 * @ingroup
 *
 * @date 2025-04-27
 * @author Jihuan Tian
 */

#include <deal.II/base/point.h>

#include <catch2/catch_all.hpp>

#include <iostream>

using namespace dealii;
using namespace Catch::Matchers;

TEST_CASE("Verify copying Point<spacedim, double> to Point<spacedim, float>",
          "[dealii]")
{
  INFO("*** test start");

  const unsigned int      spacedim = 3;
  Point<spacedim, double> p1(2.3, 3.5, 4.6);
  Point<spacedim, float>  p2;

  p2 = p1;

  std::cout << p2 << std::endl;

  INFO("*** test end");
}
