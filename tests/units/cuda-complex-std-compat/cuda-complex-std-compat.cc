// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file cuda-complex-std-compat.cc
 * @brief Unit tests ensuring the third-party cuda_complex behaves like
 * std::complex.
 * @ingroup
 *
 * @date 2026-02-07
 * @author Jihuan Tian
 */

#include <catch2/catch_all.hpp>

#include <cmath>
#include <complex>
#include <sstream>

// Third-party CUDA complex (standalone .hpp, no namespace)
#include "cuda_complex.hpp"

using namespace Catch::Matchers;

// Alias: cuda_complex.hpp defines ::complex in global namespace
template <typename T>
using CudaComplex = ::complex<T>;

constexpr double tol = 1e-15;

template <typename T>
static void
require_complex_near(const CudaComplex<T>  &cuda_z,
                     const std::complex<T> &std_z,
                     T                      rel_eps = T(tol))
{
  REQUIRE(cuda_z.real() == Catch::Approx(std_z.real()).epsilon(rel_eps));
  REQUIRE(cuda_z.imag() == Catch::Approx(std_z.imag()).epsilon(rel_eps));
}

template <typename T>
static void
require_scalar_near(T a, T b, T rel_eps = T(tol))
{
  REQUIRE(a == Catch::Approx(b).epsilon(rel_eps));
}

TEST_CASE("cuda_complex construction and accessors",
          "[cuda_complex][std_compat]")
{
  INFO("*** test start");

  SECTION("Default construction")
  {
    CudaComplex<double>  cuda_z;
    std::complex<double> std_z;
    require_complex_near(cuda_z, std_z);
  }

  SECTION("Construction from real and imaginary parts")
  {
    const double         re = 2.5, im = -1.3;
    CudaComplex<double>  cuda_z(re, im);
    std::complex<double> std_z(re, im);
    require_complex_near(cuda_z, std_z);
  }

  SECTION("Copy construction and real/imag accessors")
  {
    const double         re = 3.0, im = 4.0;
    CudaComplex<double>  cuda_z(re, im);
    CudaComplex<double>  cuda_copy(cuda_z);
    std::complex<double> std_z(re, im);
    require_complex_near(cuda_copy, std_z);
    require_scalar_near(cuda_z.real(), std_z.real());
    require_scalar_near(cuda_z.imag(), std_z.imag());
  }

  SECTION("Cross-type construction float from double")
  {
    CudaComplex<double>  cuda_d(1.5, 2.5);
    CudaComplex<float>   cuda_f(cuda_d);
    std::complex<double> std_d(1.5, 2.5);
    std::complex<float>  std_f(std_d);
    require_complex_near(cuda_f, std_f, float(tol));
  }

  SECTION("Setters real() and imag()")
  {
    CudaComplex<double>  cuda_z;
    std::complex<double> std_z;
    cuda_z.real(7.0);
    cuda_z.imag(8.0);
    std_z.real(7.0);
    std_z.imag(8.0);
    require_complex_near(cuda_z, std_z);
  }
}

TEST_CASE("cuda_complex compound assignment and arithmetic",
          "[cuda_complex][std_compat]")
{
  INFO("*** test start");

  const double re1 = 1.0, im1 = 2.0, re2 = -0.5, im2 = 1.5;

  SECTION("operator+= (complex and scalar)")
  {
    CudaComplex<double>  cuda_z(re1, im1);
    std::complex<double> std_z(re1, im1);
    cuda_z += CudaComplex<double>(re2, im2);
    std_z += std::complex<double>(re2, im2);
    require_complex_near(cuda_z, std_z);

    cuda_z += 3.0;
    std_z += 3.0;
    require_complex_near(cuda_z, std_z);
  }

  SECTION("operator-= (complex and scalar)")
  {
    CudaComplex<double>  cuda_z(re1, im1);
    std::complex<double> std_z(re1, im1);
    cuda_z -= CudaComplex<double>(re2, im2);
    std_z -= std::complex<double>(re2, im2);
    require_complex_near(cuda_z, std_z);
  }

  SECTION("operator*= (complex and scalar)")
  {
    CudaComplex<double>  cuda_z(re1, im1);
    std::complex<double> std_z(re1, im1);
    cuda_z *= CudaComplex<double>(re2, im2);
    std_z *= std::complex<double>(re2, im2);
    require_complex_near(cuda_z, std_z);

    cuda_z *= 2.0;
    std_z *= 2.0;
    require_complex_near(cuda_z, std_z);
  }

  SECTION("operator/= (complex and scalar)")
  {
    CudaComplex<double>  cuda_z(re1, im1);
    std::complex<double> std_z(re1, im1);
    CudaComplex<double>  cuda_w(re2, im2);
    std::complex<double> std_w(re2, im2);
    cuda_z /= cuda_w;
    std_z /= std_w;
    require_complex_near(cuda_z, std_z);
  }

  SECTION("Binary +, -, *, / (complex and scalar)")
  {
    CudaComplex<double>  cuda_a(re1, im1), cuda_b(re2, im2);
    std::complex<double> std_a(re1, im1), std_b(re2, im2);

    require_complex_near(cuda_a + cuda_b, std_a + std_b);
    require_complex_near(cuda_a - cuda_b, std_a - std_b);
    require_complex_near(cuda_a * cuda_b, std_a * std_b);
    require_complex_near(cuda_a / cuda_b, std_a / std_b);

    require_complex_near(cuda_a + 2.0, std_a + 2.0);
    require_complex_near(2.0 + cuda_a, 2.0 + std_a);
    require_complex_near(cuda_a - 2.0, std_a - 2.0);
    require_complex_near(2.0 - cuda_a, 2.0 - std_a);
    require_complex_near(cuda_a * 2.0, std_a * 2.0);
    require_complex_near(cuda_a / 2.0, std_a / 2.0);
  }

  SECTION("Unary + and -")
  {
    CudaComplex<double>  cuda_z(re1, im1);
    std::complex<double> std_z(re1, im1);
    require_complex_near(+cuda_z, +std_z);
    require_complex_near(-cuda_z, -std_z);
  }
}

TEST_CASE("cuda_complex comparison operators", "[cuda_complex][std_compat]")
{
  CudaComplex<double>  cuda_a(1.0, 2.0), cuda_b(1.0, 2.0), cuda_c(1.0, 3.0);
  std::complex<double> std_a(1.0, 2.0), std_b(1.0, 2.0), std_c(1.0, 3.0);

  REQUIRE((cuda_a == cuda_b) == (std_a == std_b));
  REQUIRE((cuda_a != cuda_c) == (std_a != std_c));
  REQUIRE((cuda_a == 1.0) == (std_a == 1.0));
  REQUIRE((1.0 == cuda_a) == (1.0 == std_a));
  REQUIRE((cuda_a != 2.0) == (std_a != 2.0));
}

TEST_CASE(
  "cuda_complex free functions (real, imag, abs, arg, norm, conj, polar)",
  "[cuda_complex][std_compat]")
{
  const double         re = 3.0, im = 4.0;
  CudaComplex<double>  cuda_z(re, im);
  std::complex<double> std_z(re, im);

  require_scalar_near(real(cuda_z), std::real(std_z));
  require_scalar_near(imag(cuda_z), std::imag(std_z));
  require_scalar_near(abs(cuda_z), std::abs(std_z));
  require_scalar_near(arg(cuda_z), std::arg(std_z));
  require_scalar_near(norm(cuda_z), std::norm(std_z));
  require_complex_near(conj(cuda_z), std::conj(std_z));

  SECTION("polar")
  {
    const double rho = 2.0, theta = 0.5;
    require_complex_near(polar(rho, theta), std::polar(rho, theta));
  }
}

TEST_CASE("cuda_complex transcendentals (exp, log, log10, sqrt, sin, cos, tan)",
          "[cuda_complex][std_compat]")
{
  const double         re = 0.5, im = 0.7;
  CudaComplex<double>  cuda_z(re, im);
  std::complex<double> std_z(re, im);

  require_complex_near(exp(cuda_z), std::exp(std_z));
  require_complex_near(log(cuda_z), std::log(std_z));
  require_complex_near(log10(cuda_z), std::log10(std_z));
  require_complex_near(sqrt(cuda_z), std::sqrt(std_z));
  require_complex_near(sin(cuda_z), std::sin(std_z));
  require_complex_near(cos(cuda_z), std::cos(std_z));
  require_complex_near(tan(cuda_z), std::tan(std_z));
}

TEST_CASE("cuda_complex hyperbolic (sinh, cosh, tanh)",
          "[cuda_complex][std_compat]")
{
  const double         re = 0.3, im = -0.2;
  CudaComplex<double>  cuda_z(re, im);
  std::complex<double> std_z(re, im);

  require_complex_near(sinh(cuda_z), std::sinh(std_z));
  require_complex_near(cosh(cuda_z), std::cosh(std_z));
  require_complex_near(tanh(cuda_z), std::tanh(std_z));
}

TEST_CASE("cuda_complex pow", "[cuda_complex][std_compat]")
{
  const double         re = 0.5, im = 0.3;
  CudaComplex<double>  cuda_z(re, im), cuda_w(0.2, 0.4);
  std::complex<double> std_z(re, im), std_w(0.2, 0.4);

  require_complex_near(pow(cuda_z, cuda_w), std::pow(std_z, std_w));
  require_complex_near(pow(cuda_z, 2.0), std::pow(std_z, 2.0));
  require_complex_near(pow(2.0, cuda_z), std::pow(2.0, std_z));
}

TEST_CASE("cuda_complex float specialization", "[cuda_complex][std_compat]")
{
  const float         re = 1.5f, im = -2.5f;
  CudaComplex<float>  cuda_z(re, im);
  std::complex<float> std_z(re, im);

  require_complex_near(cuda_z, std_z, 1e-5f);
  require_complex_near(cuda_z * cuda_z, std_z * std_z, 1e-5f);
  require_complex_near(conj(cuda_z), std::conj(std_z), 1e-5f);
  require_scalar_near(abs(cuda_z), std::abs(std_z), 1e-5f);
}

TEST_CASE("cuda_complex stream output", "[cuda_complex][std_compat]")
{
  CudaComplex<double> cuda_z(1.5, 2.5);
  std::ostringstream  oss;
  oss << cuda_z;
  // Format is (real,imag) per the implementation
  REQUIRE(oss.str().find('1') != std::string::npos);
  REQUIRE(oss.str().find('2') != std::string::npos);
}

TEST_CASE("cuda_complex assignment from scalar", "[cuda_complex][std_compat]")
{
  CudaComplex<double>  cuda_z(1.0, 2.0);
  std::complex<double> std_z(1.0, 2.0);
  cuda_z = 5.0;
  std_z  = 5.0;
  require_complex_near(cuda_z, std_z);

  cuda_z = 0.0;
  std_z  = 0.0;
  require_complex_near(cuda_z, std_z);
}