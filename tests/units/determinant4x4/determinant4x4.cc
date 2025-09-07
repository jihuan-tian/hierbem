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
