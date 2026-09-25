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
 * @file function_parser_template_ret.h
 * @brief Function parser whose return value has a template type.
 *
 * @date 2026-09-23
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_CONFIG_FILE_FUNCTION_PARSER_TEMPLATE_RET_H_
#define HIERBEM_INCLUDE_CONFIG_FILE_FUNCTION_PARSER_TEMPLATE_RET_H_

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include <map>
#include <string>

HBEM_NS_OPEN

using namespace dealii;

/**
 * Class for the function parser which returns a value with the specified type.
 */
template <int dim, typename Number>
class FunctionParserTemplateRet : public Function<dim, Number>
{
public:
  FunctionParserTemplateRet(const unsigned int n_components = 1,
                            const double       initial_time = 0.0,
                            const double       h            = 1e-8)
    : Function<dim, Number>(n_components, initial_time)
    , parser(n_components, initial_time, h)
  {}

  void
  initialize(const std::string                   &vars,
             const std::string                   &expression,
             const std::map<std::string, double> &constants,
             const bool                           time_dependent = false)
  {
    parser.initialize(vars, expression, constants, time_dependent);
  }

  virtual Number
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    // @p FunctionParser::value always returns double. Here we cast it to the
    // given type @p Number. When @p Number is @p std::complex, a double value
    // can also be converted to a complex value with a zero imaginary part.
    return Number(parser.value(p, component));
  }

private:
  FunctionParser<dim> parser;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CONFIG_FILE_FUNCTION_PARSER_TEMPLATE_RET_H_
