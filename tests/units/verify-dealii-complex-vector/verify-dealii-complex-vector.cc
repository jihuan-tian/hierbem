/**
 * @file verify-dealii-complex-vector.cc
 * @brief Verify complex valued Vector class in deal.ii
 *
 * @ingroup linalg
 * @author Jihuan Tian
 * @date 2025-03-01
 */
#include <deal.II/lac/vector.h>

#include <catch2/catch_all.hpp>
#include <julia.h>

#include <cmath>
#include <complex>
#include <iostream>

#include "debug_tools.h"

using namespace Catch::Matchers;
using namespace dealii;
using namespace HierBEM;

// The constructor and reinit function should initialize a vector with zeros.
void
vector_initialization()
{
  const unsigned int           n = 10;
  Vector<std::complex<double>> v1(n);

  // Ensure the vector are constructed and initialized to zero.
  for (unsigned int i = 0; i < n; i++)
    {
      REQUIRE(v1(i).real() == 0.0);
      REQUIRE(v1(i).imag() == 0.0);
    }

  for (unsigned int i = 0; i < n; i++)
    {
      v1(i).real(std::sin(i));
      v1(i).imag(std::cos(i));
    }

  // After reinitialization, the vector should be reset to zero.
  v1.reinit(n);
  for (unsigned int i = 0; i < n; i++)
    {
      REQUIRE(v1(i).real() == 0.0);
      REQUIRE(v1(i).imag() == 0.0);
    }
}

// Compute v1 = v1 + a*v2
void
vector_add()
{
  const unsigned int           n = 10;
  Vector<std::complex<double>> v1(n);
  Vector<std::complex<double>> v2(n);

  for (unsigned int i = 0; i < n; i++)
    {
      v1(i).real(i);
      v1(i).imag(i * 1.5);

      v2(i).real(std::sin(i));
      v2(i).imag(std::cos(i));
    }

  v1.add(std::complex<double>{1.0, 0.5}, v2);

  // Get the Julia variable as an array.
  jl_array_t *v3_jl = (jl_array_t *)jl_eval_string("v3");
  REQUIRE(jl_array_nrows(v3_jl) == n);
  // Extract the array data and compare with the C++ results.
  std::complex<double> *v3 = (std::complex<double> *)jl_array_data(v3_jl);
  for (unsigned int i = 0; i < n; i++)
    {
      REQUIRE(v3[i].real() == Catch::Approx(v1(i).real()).epsilon(1e-15));
      REQUIRE(v3[i].imag() == Catch::Approx(v1(i).imag()).epsilon(1e-15));
    }
}

void
vector_inner_product()
{
  const unsigned int           n = 10;
  Vector<std::complex<double>> v1(n);
  Vector<std::complex<double>> v2_complex_double(n);
  Vector<std::complex<float>>  v2_complex_float(n);

  for (unsigned int i = 0; i < n; i++)
    {
      v1(i).real(std::sin(i + 1));
      v1(i).imag(std::cos(i + 1));

      v2_complex_double(i).real(i + 1);
      v2_complex_double(i).imag(i + 1.5);

      v2_complex_float(i).real(i + 1);
      v2_complex_float(i).imag(i + 1.5);
    }

  std::complex<double> prod_with_complex_double = v1 * v2_complex_double;
  std::complex<double> prod_with_complex_float  = v1 * v2_complex_float;

  jl_value_t *ret            = jl_eval_string("real(v1_dot_v2)");
  double      v1_dot_v2_real = jl_unbox_float64(ret);
  ret                        = jl_eval_string("imag(v1_dot_v2)");
  double v1_dot_v2_imag      = jl_unbox_float64(ret);

  REQUIRE(v1_dot_v2_real ==
          Catch::Approx(prod_with_complex_double.real()).epsilon(1e-15));
  REQUIRE(v1_dot_v2_imag ==
          Catch::Approx(prod_with_complex_double.imag()).epsilon(1e-15));
  REQUIRE(v1_dot_v2_real ==
          Catch::Approx(prod_with_complex_float.real()).epsilon(1e-15));
  REQUIRE(v1_dot_v2_imag ==
          Catch::Approx(prod_with_complex_float.imag()).epsilon(1e-15));
}

void
vector_add_and_dot()
{
  const unsigned int           n = 10;
  Vector<std::complex<double>> v1(n);
  Vector<std::complex<double>> v2(n);
  Vector<std::complex<double>> v3(n);

  for (unsigned int i = 0; i < n; i++)
    {
      v1(i).real(std::sin(i + 1));
      v1(i).imag(std::cos(i + 1));

      v2(i).real(i + 1);
      v2(i).imag(i + 1.5);

      v3(i).real(std::tan(i + 1));
      v3(i).imag(std::sqrt(i + 1));
    }

  std::complex<double> add_and_dot =
    v1.add_and_dot(std::complex<double>{1.0, 0.5}, v2, v3);

  jl_value_t *ret              = jl_eval_string("real(add_and_dot)");
  double      add_and_dot_real = jl_unbox_float64(ret);
  ret                          = jl_eval_string("imag(add_and_dot)");
  double add_and_dot_imag      = jl_unbox_float64(ret);

  REQUIRE(add_and_dot_real == Catch::Approx(add_and_dot.real()).epsilon(1e-15));
  REQUIRE(add_and_dot_imag == Catch::Approx(add_and_dot.imag()).epsilon(1e-15));
}

TEST_CASE("Verify complex valued dealii::Vector", "[linalg]")
{
  INFO("*** test start");
  jl_init();
  (void)jl_eval_string("include(\"process.jl\")");

  vector_initialization();
  vector_add();
  vector_inner_product();
  vector_add_and_dot();

  jl_atexit_hook(0);
  INFO("*** test end");
}
