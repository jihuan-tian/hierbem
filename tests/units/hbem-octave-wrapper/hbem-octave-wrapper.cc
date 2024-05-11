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
