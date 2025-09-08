// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file general_exceptions.h
 * @brief Definition of self-defined general exceptions.
 * @date 2021-06-10
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_EXCEPTIONS_GENERAL_EXCEPTIONS_H_
#define HIERBEM_INCLUDE_EXCEPTIONS_GENERAL_EXCEPTIONS_H_


#include <deal.II/base/exceptions.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

DeclException3(ExcOpenIntervalRange,
               double,
               double,
               double,
               << arg1 << " is not in the range (" << arg2 << ", " << arg3
               << ").");

DeclException3(ExcLeftOpenIntervalRange,
               double,
               double,
               double,
               << arg1 << " is not in the range (" << arg2 << ", " << arg3
               << "].");

DeclException3(ExcRightOpenIntervalRange,
               double,
               double,
               double,
               << arg1 << " is not in the range [" << arg2 << ", " << arg3
               << ").");

DeclException3(ExcClosedIntervalRange,
               double,
               double,
               double,
               << arg1 << " is not in the range [" << arg2 << ", " << arg3
               << "].");

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_EXCEPTIONS_GENERAL_EXCEPTIONS_H_
