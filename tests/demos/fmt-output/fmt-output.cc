// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file fmt-output.cc
 * @brief Demostration for using fmt library in Catch2 testing framework
 *
 * @ingroup test_cases
 * @author
 * @date 2024-05-15
 */

#include <catch2/catch_all.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace Catch::Matchers;

TEST_CASE("Format string using fmt", "[fmt][demo]")
{
  {
    auto s = fmt::format("The answer is {:.3f}", 2.718281828);
    REQUIRE(s == "The answer is 2.718");
  }
  {
    auto s =
      fmt::format("Tuple: {}", std::tuple<char, int, float>{'a', 1, 2.5f});
    REQUIRE(s == "Tuple: ('a', 1, 2.5)");
  }
  {
    auto s = fmt::format("Vector: {}", std::vector<int>{1, 2, 3});
    REQUIRE(s == "Vector: [1, 2, 3]");
  }
  {
    auto s = fmt::format("Map: {}",
                         std::map<int, std::string>{{1, "one"}, {2, "two"}});
    REQUIRE(s == "Map: {1: \"one\", 2: \"two\"}");
  }
  {
    auto s = fmt::format("{:*^30}", "center aligned");
    REQUIRE(s == "********center aligned********");
  }
}
