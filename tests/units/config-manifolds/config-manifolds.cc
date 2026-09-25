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
 * @file config-manifolds.cc
 * @brief Load BEM manifold configs from an example TOML file, then write the
 * config back and compare with the example.
 *
 * @ingroup test_cases
 * @date 2026-09-18
 * @author Jihuan Tian
 */

#include <catch2/catch_all.hpp>
#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "config_file/config_file.h"

using namespace Catch::Matchers;
using namespace HierBEM;

struct ConfManifoldsExample
{
  std::vector<ConfManifold> manifolds;
};

std::string
read_file(const std::string &path)
{
  std::ifstream in(path);
  REQUIRE(in.good());
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

TEST_CASE("Read BEM manifolds from example TOML and round-trip",
          "[config][toml]")
{
  const std::string example_path(SOURCE_DIR "/example_manifolds.toml");

  const auto config =
    rfl::toml::load<ConfManifoldsExample>(example_path).value();

  REQUIRE(config.manifolds.size() == 8);

  {
    REQUIRE(rfl::holds_alternative<ConfSphericalManifold>(
      config.manifolds[0].variant()));
    const auto &m =
      rfl::get<ConfSphericalManifold>(config.manifolds[0].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{1, 2});
    REQUIRE(m.center == dealii::Point<3, double>(0.0, 0.0, 0.0));
    REQUIRE(m.mapping_order == 2);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfCylindricalManifoldCartesianAxis>(
      config.manifolds[1].variant()));
    const auto &m = rfl::get<ConfCylindricalManifoldCartesianAxis>(
      config.manifolds[1].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{3});
    REQUIRE(m.axis == 2);
    REQUIRE(m.tolerance == 1e-10);
    REQUIRE(m.mapping_order == 2);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfCylindricalManifold>(
      config.manifolds[2].variant()));
    const auto &m =
      rfl::get<ConfCylindricalManifold>(config.manifolds[2].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{4, 5});
    REQUIRE(m.direction[0] == 0.0);
    REQUIRE(m.direction[1] == 0.0);
    REQUIRE(m.direction[2] == 1.0);
    REQUIRE(m.point_on_axis == dealii::Point<3, double>(1.0, 2.0, 3.0));
    REQUIRE(m.tolerance == 1e-10);
    REQUIRE(m.mapping_order == 3);
  }

  {
    REQUIRE(
      rfl::holds_alternative<ConfTorusManifold>(config.manifolds[3].variant()));
    const auto &m = rfl::get<ConfTorusManifold>(config.manifolds[3].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{6});
    REQUIRE(m.R == 2.0);
    REQUIRE(m.r == 0.5);
    REQUIRE(m.mapping_order == 2);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfOCCNormalProjectionManifold>(
      config.manifolds[4].variant()));
    const auto &m =
      rfl::get<ConfOCCNormalProjectionManifold>(config.manifolds[4].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{7});
    REQUIRE(m.tolerance == 1e-7);
    REQUIRE(m.mapping_order == 2);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfOCCDirectionalProjectionManifold>(
      config.manifolds[5].variant()));
    const auto &m = rfl::get<ConfOCCDirectionalProjectionManifold>(
      config.manifolds[5].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{8});
    REQUIRE(m.direction[0] == 1.0);
    REQUIRE(m.direction[1] == 0.0);
    REQUIRE(m.direction[2] == 0.0);
    REQUIRE(m.tolerance == 1e-7);
    REQUIRE(m.mapping_order == 2);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfOCCNormalToMeshProjectionManifold>(
      config.manifolds[6].variant()));
    const auto &m = rfl::get<ConfOCCNormalToMeshProjectionManifold>(
      config.manifolds[6].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{9});
    REQUIRE(m.tolerance == 1e-7);
    REQUIRE(m.mapping_order == 1);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfOCCNURBSPatchManifold>(
      config.manifolds[7].variant()));
    const auto &m =
      rfl::get<ConfOCCNURBSPatchManifold>(config.manifolds[7].variant());
    REQUIRE(m.surface_tags == std::vector<EntityTag>{10, 11});
    REQUIRE(m.tolerance == 1e-7);
    REQUIRE(m.mapping_order == 2);
  }

  const std::string written = rfl::toml::write(config);
  const std::string example = read_file(example_path);
  INFO("written config:\n" << written);
  INFO("example file:\n" << example);
  REQUIRE(written == example);
}
