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
    int          out = val.int_value();
    REQUIRE(out == 3);
  }

  SECTION("test source_file()")
  {
    inst.source_file(SOURCE_DIR "/test.m");

    auto val = inst.eval_string("d");
    double       out = val.double_value();
    REQUIRE_THAT(out, WithinRel(1.0, 1e-6) || WithinAbs(1.0, 1e-8));
  }
}
