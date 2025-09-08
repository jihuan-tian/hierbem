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
 * @file preconditioner_type.h
 * @brief Definition of preconditioner types
 * @ingroup preconditioners
 *
 * @date 2025-09-01
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_TYPE_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_TYPE_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * Enum for types of preconditioners.
 */
enum PreconditionerType
{
  HMatrixFactorization,
  OperatorPreconditioning,
  Identity,
  Jacobi
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_TYPE_H_
