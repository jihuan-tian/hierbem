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
 * @file reflectcpp-usage.cc
 * @brief
 *
 * @ingroup test_cases
 * @author
 * @date 2024-05-23
 */
#include <catch2/catch_all.hpp>

// XXX This is a workaround to suppress warnings from reflect-cpp headers
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wunused-parameter"
// #pragma GCC diagnostic ignored "-Wtype-limits"
// #pragma GCC diagnostic ignored "-Wmissing-braces"
#include <rfl.hpp>
#include <rfl/toml.hpp>
// #pragma GCC diagnostic pop

#include <cstdint>
#include <optional>
#include <string>

#include "hbem_test_config.h"
#include "hbem_test_utils.h"

using namespace Catch::Matchers;

struct ConfigProject
{
  std::string                name        = "";
  std::optional<std::string> working_dir = ".";
};

struct ConfigBEM
{
  std::optional<std::uint32_t> order_dirichlet         = 1;
  std::optional<std::uint32_t> order_neumann           = 0;
  std::optional<std::uint32_t> mapping_order_dirichlet = 1;
  std::optional<std::uint32_t> mapping_order_neumann   = 1;
  std::optional<bool>          is_interior_problem     = false;
};

struct ConfigHierBEM
{
  ConfigProject            project = ConfigProject{};
  std::optional<ConfigBEM> bem     = ConfigBEM{};
};

TEST_CASE("TOML serialize", "[toml][demo]")
{
  const auto config = ConfigHierBEM{
    .project = ConfigProject{.name = "test", .working_dir = "output"}};
  const std::string toml_str = rfl::toml::write(config);
  rfl::toml::save("serialized.toml", config);

  REQUIRE_THAT(toml_str, ContainsSubstring("name = 'test'"));
  REQUIRE_THAT(toml_str, ContainsSubstring("working_dir = 'output'"));
  REQUIRE_THAT(toml_str, ContainsSubstring("order_dirichlet = 1"));
  REQUIRE_THAT(toml_str, ContainsSubstring("order_neumann = 0"));
  REQUIRE_THAT(toml_str, ContainsSubstring("mapping_order_dirichlet = 1"));
  REQUIRE_THAT(toml_str, ContainsSubstring("mapping_order_neumann = 1"));
  REQUIRE_THAT(toml_str, ContainsSubstring("is_interior_problem = false"));
}

TEST_CASE("TOML deserialize", "[toml][demo]")
{
  const auto default_conf = ConfigHierBEM{};
  const auto extern_conf =
    rfl::toml::load<ConfigHierBEM>(SOURCE_DIR "/test_config.toml").value();

  const auto actual_conf = extern_conf;
  // rfl::replace<ConfigHierBEM>(default_conf, extern_conf);

  REQUIRE(actual_conf.project.name == "test");
  REQUIRE(actual_conf.project.working_dir.value_or(".") == "output");

  if (actual_conf.bem)
    {
      REQUIRE(actual_conf.bem->order_dirichlet.value_or(1) == 1);
      REQUIRE(actual_conf.bem->order_neumann.value_or(0) == 0);
      REQUIRE(actual_conf.bem->mapping_order_dirichlet.value_or(1) == 1);
      REQUIRE(actual_conf.bem->mapping_order_neumann.value_or(1) == 1);
      REQUIRE(actual_conf.bem->is_interior_problem.value_or(false) == true);
    }
}
