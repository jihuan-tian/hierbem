// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include "config_file/config_file.h"

#include <rfl.hpp>

#include <type_traits>
#include <variant>

#include "config.h"

HBEM_NS_OPEN

bool
is_complex_value_spec(const ConfValueSpec &spec)
{
  return spec.visit([](const auto &alternative) -> bool {
    using T = std::decay_t<decltype(alternative)>;
    if constexpr (std::is_same_v<T, ConfConstantValue>)
      return rfl::holds_alternative<std::complex<double>>(alternative.value);
    else if constexpr (std::is_same_v<T, ConfExpressionComplex>)
      return true;
    else
      return false;
  });
}

bool
is_solver_complex_valued()
{
  const auto &conf_inst = ConfigFile::instance().getConfig();
  return conf_inst.boundary_conditions[0].visit(
    [](const auto &alternative) -> bool {
      return is_complex_value_spec(alternative.value_spec);
    });
}

HBEM_NS_CLOSE
