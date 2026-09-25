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
 * @file config-boundary-conditions.cc
 * @brief Load BEM boundary conditions from an example TOML file, then write
 * the config back and compare with the example.
 *
 * @ingroup test_cases
 * @date 2026-09-18
 * @author Jihuan Tian
 */

#include <catch2/catch_all.hpp>
#include <rfl.hpp>
#include <rfl/toml.hpp>

#include <complex>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "config_file/config_file.h"

using namespace Catch::Matchers;
using namespace HierBEM;

struct ConfBoundaryConditionsExample
{
  std::vector<ConfBEMBoundaryCondition> boundary_conditions;
};

std::string
read_file(const std::string &path)
{
  std::ifstream in(path);
  REQUIRE(in.good());
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

TEST_CASE("Read BEM boundary conditions from example TOML and round-trip",
          "[config][toml]")
{
  const std::string example_path(SOURCE_DIR
                                 "/example_boundary_conditions.toml");

  const auto config =
    rfl::toml::load<ConfBoundaryConditionsExample>(example_path).value();

  REQUIRE(config.boundary_conditions.size() == 6);

  {
    REQUIRE(rfl::holds_alternative<ConfBEMDirichletBC>(
      config.boundary_conditions[0].variant()));
    const auto &bc =
      rfl::get<ConfBEMDirichletBC>(config.boundary_conditions[0].variant());
    REQUIRE(bc.surface_tags == std::vector<EntityTag>{1, 2});
    REQUIRE(rfl::holds_alternative<ConfConstantValue>(bc.value_spec.variant()));
    const auto &value_spec =
      rfl::get<ConfConstantValue>(bc.value_spec.variant());
    REQUIRE(rfl::holds_alternative<double>(value_spec.value));
    REQUIRE(rfl::get<double>(value_spec.value) == 1.0);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfBEMDirichletBC>(
      config.boundary_conditions[1].variant()));
    const auto &bc =
      rfl::get<ConfBEMDirichletBC>(config.boundary_conditions[1].variant());
    REQUIRE(bc.surface_tags == std::vector<EntityTag>{3});
    REQUIRE(rfl::holds_alternative<ConfExpression>(bc.value_spec.variant()));
    const auto &value_spec = rfl::get<ConfExpression>(bc.value_spec.variant());
    REQUIRE(value_spec.expr == "sin(x) * cos(z)");
  }

  {
    REQUIRE(rfl::holds_alternative<ConfBEMDirichletBC>(
      config.boundary_conditions[2].variant()));
    const auto &bc =
      rfl::get<ConfBEMDirichletBC>(config.boundary_conditions[2].variant());
    REQUIRE(bc.surface_tags == std::vector<EntityTag>{4});
    REQUIRE(rfl::holds_alternative<ConfConstantValue>(bc.value_spec.variant()));
    const auto &value_spec =
      rfl::get<ConfConstantValue>(bc.value_spec.variant());
    REQUIRE(rfl::holds_alternative<std::complex<double>>(value_spec.value));
    const auto value = rfl::get<std::complex<double>>(value_spec.value);
    REQUIRE(value.real() == 0.0);
    REQUIRE(value.imag() == 1.0);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfBEMNeumannBC>(
      config.boundary_conditions[3].variant()));
    const auto &bc =
      rfl::get<ConfBEMNeumannBC>(config.boundary_conditions[3].variant());
    REQUIRE(bc.surface_tags == std::vector<EntityTag>{5, 6});
    REQUIRE(
      rfl::holds_alternative<ConfExpressionComplex>(bc.value_spec.variant()));
    const auto &value_spec =
      rfl::get<ConfExpressionComplex>(bc.value_spec.variant());
    REQUIRE(value_spec.real_part_expr == "cos(x)");
    REQUIRE(value_spec.imag_part_expr == "sin(z)");
  }

  {
    REQUIRE(rfl::holds_alternative<ConfBEMRobinBC>(
      config.boundary_conditions[4].variant()));
    const auto &bc =
      rfl::get<ConfBEMRobinBC>(config.boundary_conditions[4].variant());
    REQUIRE(bc.surface_tags == std::vector<EntityTag>{7});
    REQUIRE(rfl::holds_alternative<ConfConstantValue>(bc.coeff_spec.variant()));
    const auto &coeff_spec =
      rfl::get<ConfConstantValue>(bc.coeff_spec.variant());
    REQUIRE(rfl::holds_alternative<double>(coeff_spec.value));
    REQUIRE(rfl::get<double>(coeff_spec.value) == 2.0);
    REQUIRE(rfl::holds_alternative<ConfConstantValue>(bc.value_spec.variant()));
    const auto &value_spec =
      rfl::get<ConfConstantValue>(bc.value_spec.variant());
    REQUIRE(rfl::holds_alternative<double>(value_spec.value));
    REQUIRE(rfl::get<double>(value_spec.value) == 0.0);
  }

  {
    REQUIRE(rfl::holds_alternative<ConfBEMRobinBC>(
      config.boundary_conditions[5].variant()));
    const auto &bc =
      rfl::get<ConfBEMRobinBC>(config.boundary_conditions[5].variant());
    REQUIRE(bc.surface_tags == std::vector<EntityTag>{10, 11});
    REQUIRE(rfl::holds_alternative<ConfConstantValue>(bc.coeff_spec.variant()));
    const auto &coeff_spec =
      rfl::get<ConfConstantValue>(bc.coeff_spec.variant());
    REQUIRE(rfl::holds_alternative<std::complex<double>>(coeff_spec.value));
    const auto value = rfl::get<std::complex<double>>(coeff_spec.value);
    REQUIRE(value.real() == 2.0);
    REQUIRE(value.imag() == 3.0);
    REQUIRE(
      rfl::holds_alternative<ConfExpressionComplex>(bc.value_spec.variant()));
    const auto &value_spec =
      rfl::get<ConfExpressionComplex>(bc.value_spec.variant());
    REQUIRE(value_spec.real_part_expr == "tan(x)");
    REQUIRE(value_spec.imag_part_expr == "ctan(x)");
  }

  const std::string written = rfl::toml::write(config);
  const std::string example = read_file(example_path);
  INFO("written config:\n" << written);
  INFO("example file:\n" << example);
  REQUIRE(written == example);
}
