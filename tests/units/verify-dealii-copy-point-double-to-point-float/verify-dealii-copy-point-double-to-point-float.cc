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
