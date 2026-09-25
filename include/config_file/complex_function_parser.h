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
 * @file complex_function_parser.h
 * @brief Complex valued function parser.
 *
 * @date 2026-09-21
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_CONFIG_FILE_COMPLEX_FUNCTION_PARSER_H_
#define HIERBEM_INCLUDE_CONFIG_FILE_COMPLEX_FUNCTION_PARSER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>

#include <complex>
#include <map>
#include <string>

#include "config.h"
#include "utilities/concepts.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Class for complex valued function parser, which is based on the muparser
 * library.
 */
template <int dim, HostComplex Number>
class ComplexFunctionParser : public Function<dim, Number>
{
public:
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  ComplexFunctionParser(const unsigned int n_components = 1,
                        const double       initial_time = 0.0,
                        const double       h            = 1e-8)
    : Function<dim, Number>(n_components, initial_time)
    , real_part_func(n_components, initial_time, h)
    , imag_part_func(n_components, initial_time, h)
  {
    Assert(n_components == 1, ExcMessage("Only one component is allowed"));
  }

  /**
   * Initialize function parsers for the real part and imaginary part
   * respectively.
   */
  void
  initialize(const std::string                   &vars,
             const std::string                   &real_part_expr,
             const std::string                   &imag_part_expr,
             const std::map<std::string, double> &constants,
             const bool                           time_dependent = false)
  {
    real_part_func.initialize(vars, real_part_expr, constants, time_dependent);
    imag_part_func.initialize(vars, imag_part_expr, constants, time_dependent);
  }

  virtual Number
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    Assert(component == 0, ExcMessage("Only one component is allowed"));

    return Number(static_cast<real_type>(real_part_func.value(p, component)),
                  static_cast<real_type>(imag_part_func.value(p, component)));
  }

private:
  FunctionParser<dim> real_part_func;
  FunctionParser<dim> imag_part_func;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CONFIG_FILE_COMPLEX_FUNCTION_PARSER_H_
