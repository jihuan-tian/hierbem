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
 * @file solver_bicgstab_general.h
 * @brief Bicgstab solver which can handle complex values.
 * @ingroup linalg
 *
 * @date 2025-05-10
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_SOLVERS_SOLVER_BICGSTAB_GENERAL_H_
#define HIERBEM_INCLUDE_SOLVERS_SOLVER_BICGSTAB_GENERAL_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_control.h>

#include <type_traits>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

template <typename VectorType>
class SolverBicgstabGeneral
{
public:
  using size_type = std::make_unsigned<types::blas_int>::type;
  using Number    = typename VectorType::value_type;

  SolverBicgstabGeneral(SolverControl &cn)
    : control(cn)
  {}

  ~SolverBicgstabGeneral() = default;

  /**
   * @param x Solution vector, its initial values will be used for computing the
   * initial residual vector.
   */
  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType         &A,
        VectorType               &x,
        const VectorType         &b,
        const PreconditionerType &preconditioner);

private:
  SolverControl &control;
};


template <typename VectorType>
template <typename MatrixType, typename PreconditionerType>
void
SolverBicgstabGeneral<VectorType>::solve(
  const MatrixType         &A,
  VectorType               &x,
  const VectorType         &b,
  const PreconditionerType &preconditioner)
{
  if constexpr (numbers::NumberTraits<Number>::is_complex)
    {
      Assert(false, ExcNotImplemented());
    }
  else
    {
      SolverBicgstab<VectorType> solver(control);
      solver.solve(A, x, b, preconditioner);
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SOLVERS_SOLVER_BICGSTAB_GENERAL_H_
