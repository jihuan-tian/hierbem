/**
 * @file call-octave.cc
 * @brief
 *
 * @ingroup testers
 * @author
 * @date 2023-10-08
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <octave/builtin-defun-decls.h>
#include <octave/interpreter.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

#include <iostream>

using namespace Catch::Matchers;

TEST_CASE("Call Octave gcd() function from C++", "[octave]")
{
  // Create interpreter.
  octave::interpreter interpreter;

  int status = interpreter.execute();
  REQUIRE(status == 0);

  octave_idx_type   n = 2;
  octave_value_list in;

  for (octave_idx_type i = 0; i < n; i++)
    in(i) = octave_value(5 * (i + 2));

  octave_value_list out = octave::feval("gcd", in, 1);
  REQUIRE(out.length() > 0);
  REQUIRE(out(0).int_value() == 5);
}

TEST_CASE("Run external Octave .m file from C++", "[octave]")
{
  // Create interpreter.
  octave::interpreter interpreter;

  int status = interpreter.execute();
  REQUIRE(status == 0);

  octave::source_file(std::string(SOURCE_DIR "/test.m"));

  int               parse_status;
  octave_value_list out = octave::eval_string("mat2str(a)", true, parse_status);
  REQUIRE(out.length() > 0);
  REQUIRE(out(0).string_value() == "[1 2;3 4]");
}
