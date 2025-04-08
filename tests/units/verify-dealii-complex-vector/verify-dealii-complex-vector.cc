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

#include <cmath>
#include <complex>

#include "hbem_julia_cpp_compare.h"
#include "hbem_julia_wrapper.h"

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
  compare_with_jl_array(v1, "v3", 1e-15, 1e-15);
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

  compare_with_jl_scalar(prod_with_complex_double.real(),
                         "real(v1_dot_v2)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(prod_with_complex_double.imag(),
                         "imag(v1_dot_v2)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(prod_with_complex_float.real(),
                         "real(v1_dot_v2)",
                         1e-15,
                         1e-15);
  compare_with_jl_scalar(prod_with_complex_float.imag(),
                         "imag(v1_dot_v2)",
                         1e-15,
                         1e-15);
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

  compare_with_jl_scalar(add_and_dot.real(), "real(add_and_dot)", 1e-15, 1e-15);
  compare_with_jl_scalar(add_and_dot.imag(), "imag(add_and_dot)", 1e-15, 1e-15);
}

TEST_CASE("Verify complex valued dealii::Vector", "[linalg]")
{
  INFO("*** test start");
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();
  inst.source_file("process.jl");

  vector_initialization();
  vector_add();
  vector_inner_product();
  vector_add_and_dot();

  INFO("*** test end");
}
