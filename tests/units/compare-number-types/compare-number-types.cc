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
 * @file compare-number-types.cc
 * @brief Compare two number types.
 * @ingroup
 *
 * @date 2025-04-08
 * @author Jihuan Tian
 */

#include <boost/type_index.hpp>

#include <catch2/catch_all.hpp>

#include <complex>
#include <fstream>
#include <iostream>

#include "hbem_cpp_validate.h"
#include "utilities/number_traits.h"

using namespace Catch::Matchers;
using namespace HierBEM;

template <typename Number1, typename Number2>
void
compare_two_types(std::ostream &out)
{
  out << "=== " << boost::typeindex::type_id<Number1>().pretty_name() << ", "
      << boost::typeindex::type_id<Number2>().pretty_name() << " ===\n";

  if (is_number_comparable<Number1, Number2>())
    out << std::boolalpha << ">: " << is_number_larger<Number1, Number2>()
        << "\n>=: " << is_number_larger_or_equal<Number1, Number2>()
        << "\n<: " << is_number_smaller<Number1, Number2>()
        << "\n<=: " << is_number_smaller_or_equal<Number1, Number2>()
        << std::endl;
  else
    out << "not comparable" << std::endl;
}

TEST_CASE("Compare two number types", "[type]")
{
  INFO("*** test start");

  std::ofstream ofs("compare-number-types.log");

  compare_two_types<float, float>(ofs);
  compare_two_types<double, double>(ofs);
  compare_two_types<std::complex<float>, std::complex<float>>(ofs);
  compare_two_types<std::complex<double>, std::complex<double>>(ofs);

  compare_two_types<float, double>(ofs);
  compare_two_types<double, float>(ofs);
  compare_two_types<float, std::complex<float>>(ofs);
  compare_two_types<std::complex<float>, float>(ofs);
  compare_two_types<float, std::complex<double>>(ofs);
  compare_two_types<std::complex<double>, float>(ofs);
  compare_two_types<double, std::complex<float>>(ofs);
  compare_two_types<std::complex<float>, double>(ofs);
  compare_two_types<double, std::complex<double>>(ofs);
  compare_two_types<std::complex<double>, double>(ofs);

  ofs.close();

  auto check_equality = [](const auto &a, const auto &b) {
    INFO("Operand 1: " << a);
    INFO("Operand 2: " << b);
    REQUIRE(a == b);
  };
  compare_two_files(SOURCE_DIR "/reference.output",
                    "compare-number-types.log",
                    check_equality);
}
