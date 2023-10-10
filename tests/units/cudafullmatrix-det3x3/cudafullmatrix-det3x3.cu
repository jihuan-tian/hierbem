/**
 * @file cudafullmatrix-det3x3.cu
 * @brief Verify the calculation of the determinant of a 3x3 matrix.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-24
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>

#include "cu_fullmatrix.hcu"

using namespace HierBEM::CUDAWrappers;
using namespace Catch::Matchers;

TEST_CASE("Calculate determinant of 3x3 matrix by CUDA", "[cuda_full_matrix]")
{
  double data[9]{
    tan(1), tan(2), tan(3), tan(4), tan(5), tan(6), tan(7), tan(8), tan(9)};
  CUDAFullMatrix<double> A(data, 3, 3);

  auto   res      = A.determinant3x3();
  double expected = -0.588238;

  REQUIRE_THAT(res, WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
}
