/**
 * @file preconditioner_type.h
 * @brief Definition of preconditioner types
 * @ingroup preconditioner
 *
 * @date 2025-09-01
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_H_

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

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_H_
