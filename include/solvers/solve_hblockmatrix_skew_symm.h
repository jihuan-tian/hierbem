// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_INCLUDE_SOLVERS_SOLVE_HBLOCKMATRIX_SKEW_SYMM_H_
#define HIERBEM_INCLUDE_SOLVERS_SOLVE_HBLOCKMATRIX_SKEW_SYMM_H_

#include <deal.II/base/numbers.h>

#include <deal.II/lac/solver_control.h>

#include "solvers/schur_complement.h"
#include "solvers/solver_cg_general.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Solve the skew symmetric block matrix system.
 *
 * @param block_mat Skew symmetric block matrix having the structure
 * \f[
 * \begin{pmatrix}
 * M_{11} & M_{12} \\
 * -M_{12}^{\mathrm{T}} & M_{22}
 * \end{pmatrix}.
 * \f]
 * Only three matrix blocks are stored.
 * @param x1 First part in the solution vector.
 * @param x2 Second part in the solution vector.
 * @param b1 First part in the right hand side vector.
 * @param b2 Second part in the right hand side vector.
 * @param precond11 Preconditioner for solving the matrix block \f$M_{11}\f$.
 * @param precond22 Preconditioner for solving the matrix block \f$M_{22}\f$.
 * @param solver_control Solver control object.
 */
template <typename SkewSymmetricBlockMatrixType,
          typename VectorType,
          typename PrecondM11Type,
          typename PrecondM22Type,
          typename Number>
void
solve_hblockmatrix_skew_symm_using_schur_complement(
  const SkewSymmetricBlockMatrixType &block_mat,
  VectorType                         &x1,
  VectorType                         &x2,
  const VectorType                   &b1,
  const VectorType                   &b2,
  const PrecondM11Type               &precond11,
  const PrecondM22Type               &precond22,
  const unsigned int                  max_iter,
  const Number                        tolerance,
  const bool                          log_history,
  const bool                          log_result)
{
  // Compute \f$M_{11}^{-1} b_1\f$ and @p b4 stores the result.
  SolverControl solver_control1(max_iter, tolerance, log_history, log_result);
  SolverCGGeneral<VectorType> solver1(solver_control1);
  VectorType                  b4(block_mat.get_M11().get_n());

  solver1.solve(block_mat.get_M11(), b4, b1, precond11);

  // Compute \f$M_{12}^{\mathrm{T}} M_{11}^{-1} b_1 = M_{12}^{\mathrm{T}} b_4
  // \f$. The result vector is @p b3.
  VectorType b3(block_mat.get_M12().get_n());
  block_mat.get_M12().Tvmult(b3, b4);

  // Compute the RHS vector in the equation for solving @p x2.
  b3 += b2;

  // Construct the Schur complement matrix and solve it for @p x2.
  SchurComplement<SkewSymmetricBlockMatrixType,
                  VectorType,
                  PrecondM11Type,
                  Number>
    schur_complement(
      block_mat, precond11, max_iter, tolerance, log_history, log_result);
  SolverControl solver_control2(max_iter, tolerance, log_history, log_result);
  SolverCGGeneral<VectorType> solver2(solver_control2);
  solver2.solve(schur_complement, x2, b3, precond22);

  // Compute \f$-M_{12}x_2\f$ and store it into @p b4.
  block_mat.get_M12().vmult(
    b4, typename numbers::NumberTraits<Number>::real_type(-1.0), x2);
  b4 += b1;

  // Solve @p x1.
  SolverControl solver_control3(max_iter, tolerance, log_history, log_result);
  SolverCGGeneral<VectorType> solver3(solver_control3);
  solver3.solve(block_mat.get_M11(), x1, b4, precond11);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SOLVERS_SOLVE_HBLOCKMATRIX_SKEW_SYMM_H_
