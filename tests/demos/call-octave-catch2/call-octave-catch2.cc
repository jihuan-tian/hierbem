/**
 * @file call-octave.cc
 * @brief Demostration for using Octave in Catch2 testing framework
 *
 * @ingroup testers
 * @author
 * @date 2023-10-08
 */
#include <catch2/catch_all.hpp>
#include <octave/builtin-defun-decls.h>
#include <octave/interpreter.h>
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>

using namespace Catch::Matchers;

TEST_CASE("Call builtins directly from C++", "[octave][demo]")
{
  // No interpreter instance needed for directly calling builtins

  {
    // Call builtin function norm() directly
    octave_idx_type n     = 2;
    Matrix          a_mat = Matrix(n, n);

    for (octave_idx_type i = 0; i < n; i++)
      for (octave_idx_type j = 0; j < n; j++)
        a_mat(i, j) = (i + 1) + (j + 1);

    octave_value_list in;
    in(0) = a_mat;

#if OCTAVE_MAJOR_VERSION < 8
    octave_value_list out = Fnorm(in, 1 /*n_ret*/);
#else
    octave_value_list out = octave::Fnorm(in, 1 /*n_ret*/);
#endif
    REQUIRE(out.length() == 1);

    double expected = 6.162278;
    REQUIRE_THAT(out(0).double_value(),
                 WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
  }

  {
    // Call builtin function gcd() directly
    octave_idx_type   n = 2;
    octave_value_list in;

    for (octave_idx_type i = 0; i < n; i++)
      in(i) = octave_value(5 * (i + 2));

#if OCTAVE_MAJOR_VERSION < 8
    octave_value_list out = Fgcd(in, 1 /*n_ret*/);
#else
    octave_value_list out = octave::Fgcd(in, 1 /*n_ret*/);
#endif

    REQUIRE(out.length() == 1);
    REQUIRE(out(0).int_value() == 5);
  }
}

TEST_CASE("Call functions through interpreter from C++", "[octave][demo]")
{
  // Create interpreter
  // NOTE: There can be only one Octave interpeter active per thread.
  // As each testcase is executed in a freshly new thread in Catch2,
  // we must ensure that only one interpreter is created per testcase.
  octave::interpreter interpreter;

  // Initialize and start Octave interpreter instance
  int status = interpreter.execute();
  REQUIRE(status == 0);

  {
    // Call builtin function gcd() through interpreter
    octave_idx_type   n = 2;
    octave_value_list in;

    for (octave_idx_type i = 0; i < n; i++)
      in(i) = octave_value(5 * (i + 2));

    octave_value_list out = octave::feval("gcd", in, 1 /*n_ret*/);
    REQUIRE(out.length() == 1);
    REQUIRE(out(0).int_value() == 5);
  }

  {
    // Call function cond() defined in *.m files through interpreter
    octave_idx_type n = 2;

    Matrix a_mat = Matrix(n, n);
    for (octave_idx_type i = 0; i < n; i++)
      for (octave_idx_type j = 0; j < n; j++)
        a_mat(i, j) = (i + 1) * 10 + (j + 1);

    // Stringify Octave matrix object by streaming interface
    std::ostringstream oss;
    oss << a_mat;
    std::string a_str = oss.str();

    // NOTE C++11 raw literals
    const char *exp_s = R"""( 11 12
 21 22
)""";
    REQUIRE(a_str == exp_s);

    // Calculate frobenius norm of matrix
    octave_value_list in;
    in(0) = a_mat;
    in(1) = "fro";

    octave_value_list out = octave::feval("cond", in, 1 /*n_ret*/);
    REQUIRE(out.length() == 1);

    double expected = 119.0;
    REQUIRE_THAT(out(0).double_value(),
                 WithinAbs(expected, 1e-6) || WithinRel(expected, 1e-8));
  }

#if OCTAVE_MAJOR_VERSION == 6
  // need call interpreter's shutdown() method explicitly to prevent segfault on
  // exit
  interpreter.shutdown();
#endif
}

TEST_CASE("Source external M-file from C++", "[octave][demo]")
{
  // Create interpreter
  // NOTE: There can be only one Octave interpeter active per thread.
  // As each testcase is executed in a freshly new thread in Catch2,
  // we must ensure that only one interpreter is created per testcase.
  octave::interpreter interpreter;

  int status = interpreter.execute();
  REQUIRE(status == 0);

  {
    // REQUIRE_NOTHROW macro only accept expressions but not statements
    // (void-type function, eg.). Statements must be wrapped in C++11
    // lambda function instead.
    REQUIRE_NOTHROW([&]() {
      // This may throw out exceptions
      octave::source_file(SOURCE_DIR "/test.m");
    }());

    int               parse_status;
    octave_value_list out =
      interpreter.eval_string("mat2str(a)", true, parse_status);
    REQUIRE(parse_status == 0);
    REQUIRE(out.length() > 0);
    REQUIRE(out(0).string_value() == "[11 22;33 44]");
  }

#if OCTAVE_MAJOR_VERSION == 6
  // need call interpreter's shutdown() method explicitly to prevent segfault on
  // exit
  interpreter.shutdown();
#endif
}
