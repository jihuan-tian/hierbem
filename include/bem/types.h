// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file types.h
 * @brief Definition of kernel function types of boundary integral operators in
 * BEM.
 *
 * @date 2025-12-13
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_BEM_TYPES_H_
#define HIERBEM_INCLUDE_BEM_TYPES_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * Enum for various types of kernel functions.
 */
enum KernelType
{
  SingleLayer,
  DoubleLayer,
  AdjointDoubleLayer,
  HyperSingular,
  HyperSingularRegular,
  NoneType
};

/**
 * Enum for various types of boundary conditions.
 */
enum ProblemType
{
  NeumannBCProblem,   //!< NeumannBCProblem
  DirichletBCProblem, //!< DirichletBCProblem
  MixedBCProblem,     //!< MixedBCProblem
  UndefinedProblem
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_TYPES_H_