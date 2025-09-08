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
 * @file hbem-julia-wrapper.cc
 * @brief Verify the wrapper class for calling Julia API.
 *
 * @author Jihuan Tian
 * @date 2025-03-07
 */

#include <catch2/catch_all.hpp>

#include <iostream>

#include "config.h"
#include "hbem_julia_wrapper.h"

using namespace Catch::Matchers;
using namespace HierBEM;

TEST_CASE("Verify Julia wrapper functions", "[wrapper]")
{
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();

  inst.source_file(SOURCE_DIR "/generate_data.jl");

  std::cout << "(unsigned int) a=" << inst.get_uint_var("a") << "\n";
  std::cout << "(int) b=" << inst.get_int_var("b") << "\n";
  std::cout << "(float) c=" << inst.get_float_var("c") << "\n";
  std::cout << "(double) d=" << inst.get_double_var("d") << "\n";
  std::cout << "(complex float) e=" << inst.get_complex_float_var("e") << "\n";
  std::cout << "(complex double) f=" << inst.get_complex_double_var("f")
            << "\n";

  HBEMJuliaValue val = inst.eval_string("g");
  size_t         n1  = val.nrows();
  REQUIRE(n1 == val.length());
  REQUIRE(n1 == val.size(0));
  REQUIRE(val.ndims() == 1);
  float *float_array = val.float_array();
  std::cout << "(float array) g=";
  for (size_t i = 0; i < n1; i++)
    std::cout << float_array[i] << " ";
  std::cout << "\n";

  val       = inst.eval_string("h");
  size_t n2 = val.nrows();
  REQUIRE(n2 == val.length());
  REQUIRE(n2 == val.size(0));
  REQUIRE(val.ndims() == 1);
  double *double_array = val.double_array();
  std::cout << "(double array) h=";
  for (size_t i = 0; i < n2; i++)
    std::cout << double_array[i] << " ";
  std::cout << "\n";

  val       = inst.eval_string("i");
  size_t n3 = val.nrows();
  REQUIRE(n3 == val.length());
  REQUIRE(n3 == val.size(0));
  REQUIRE(val.ndims() == 1);
  std::complex<float> *complex_float_array = val.complex_float_array();
  std::cout << "(complex float array) i=";
  for (size_t i = 0; i < n3; i++)
    std::cout << complex_float_array[i] << " ";
  std::cout << "\n";

  val       = inst.eval_string("j");
  size_t n4 = val.nrows();
  REQUIRE(n4 == val.length());
  REQUIRE(n4 == val.size(0));
  REQUIRE(val.ndims() == 1);
  std::complex<double> *complex_double_array = val.complex_double_array();
  std::cout << "(complex double array) j=";
  for (size_t i = 0; i < n4; i++)
    std::cout << complex_double_array[i] << " ";
  std::cout << "\n";

  // Get arrays g, h, i, j again using template member function.
  float_array = inst.get_array_var<float>("g");
  std::cout << "(float array) g=";
  for (size_t i = 0; i < n1; i++)
    std::cout << float_array[i] << " ";
  std::cout << "\n";

  double_array = inst.get_array_var<double>("h");
  std::cout << "(double array) h=";
  for (size_t i = 0; i < n2; i++)
    std::cout << double_array[i] << " ";
  std::cout << "\n";

  complex_float_array = inst.get_array_var<std::complex<float>>("i");
  std::cout << "(complex float array) i=";
  for (size_t i = 0; i < n3; i++)
    std::cout << complex_float_array[i] << " ";
  std::cout << "\n";

  complex_double_array = inst.get_array_var<std::complex<double>>("j");
  std::cout << "(complex double array) j=";
  for (size_t i = 0; i < n4; i++)
    std::cout << complex_double_array[i] << " ";
  std::cout << "\n";

  val = inst.eval_string("k");
  REQUIRE(val.size(0) == 3);
  REQUIRE(val.size(1) == 7);
}
