// Copyright (C) 2023 Xiaozhe Wang <chaoslawful@gmail.com>
// Copyright (C) 2024 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file hbem-octave-wrapper.cc
 * \brief Tests for Octave wrapper
 *
 * \author Xiaozhe Wang
 * \date 2023-10-12
 */

#include <catch2/catch_all.hpp>

#include "hbem_octave_wrapper.h"

using namespace Catch::Matchers;
using namespace HierBEM;

TEST_CASE("test octave wrapper functionality", "[wrapper]")
{
  HBEMOctaveWrapper &inst = HBEMOctaveWrapper::get_instance();

  SECTION("test single retval eval_string()")
  {
    auto val = inst.eval_string("a=1;\nb=2;\na+b");
    int  out = val.int_value();
    REQUIRE(out == 3);
  }

  SECTION("test function without return value eval_function_void()")
  {
    inst.add_path(SOURCE_DIR);
    inst.eval_function_void("func_without_ret();");
  }

  SECTION("test function with a single return value eval_function_scalar()")
  {
    inst.add_path(SOURCE_DIR);
    auto val = inst.eval_function_scalar("factorial(5);");
    int  out = val.int_value();
    REQUIRE(out == 120);
  }

  SECTION("test function with multiple return values eval_function()")
  {
    inst.add_path(SOURCE_DIR);
    HBEMOctaveValueList vals;
    inst.eval_function("[a, b] = func_multiple_rets()", 2, vals);
    int out = vals[0].int_value();
    REQUIRE(out == 3);
    out = vals[1].int_value();
    REQUIRE(out == 4);
  }

  SECTION("test source_file()")
  {
    inst.source_file(SOURCE_DIR "/test.m");

    auto   val = inst.eval_string("d");
    double out = val.double_value();
    REQUIRE_THAT(out, WithinRel(1.0, 1e-6) || WithinAbs(1.0, 1e-8));
  }
}
