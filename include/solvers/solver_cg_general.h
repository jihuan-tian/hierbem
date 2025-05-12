/**
 * @file solver_cg_general.h
 * @brief CG solver which can handle complex values.
 * @ingroup linalg
 *
 * @date 2025-05-10
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_SOLVERS_SOLVER_CG_GENERAL_H_
#define HIERBEM_INCLUDE_SOLVERS_SOLVER_CG_GENERAL_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/solver_control.h>

#include <iostream>

#include "blas_helpers.h"
#include "config.h"
#include "linalg.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * General preconditioned CG solver class, which can handle complex values.
 *
 * At the moment, this is a quick implementation, which does not inherit from
 * @p dealii::SolverBase.
 */
template <typename VectorType>
class SolverCGGeneral
{
public:
  using size_type = types::blas_int;
  using Number    = typename VectorType::value_type;

  SolverCGGeneral(SolverControl &cn)
    : control(cn)
  {}

  ~SolverCGGeneral() = default;

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
SolverCGGeneral<VectorType>::solve(const MatrixType         &A,
                                   VectorType               &x,
                                   const VectorType         &b,
                                   const PreconditionerType &preconditioner)
{
  // 2025-05-10: At the moment, @p LAPACKFullMatrixExt uses @p m() and @p n() to
  // get matrix sizes while @p HMaxtrix uses @p get_m() and @p get_n(). Because
  // they are not consistent, we do not perform a dimension assertion about the
  // matrix here.
  AssertDimension(x.size(), b.size());
  LogStream::Prefix prefix("cg-general");

  const size_type n = b.size();

  // Compute the residual vector \f$Ax^0-b\f$.
  VectorType r(b);
  // TODO Vector scaling should be generalized to handle more vector types.
  BLASHelpers::scal_helper(n, Number(-1.0), r);
  A.vmult_add(r, Number(1.0), x);

  // Preconditioned residual vector.
  VectorType v(n);
  preconditioner.vmult(v, r);

  // Correction vector
  VectorType p(v);
  VectorType p_prev(p);

  Number rho0     = LinAlg::inner_product_tbb(v, r);
  Number rho_prev = rho0;
  Number rho      = rho0;

  if (control.log_result())
    {
      deallog << "step=0,residual_norm="
              << numbers::NumberTraits<Number>::abs(rho0) << std::endl;
    }

  unsigned int i = 1;
  for (; i <= control.max_steps(); i++)
    {
      // Transformed correction vector.
      VectorType s(n);
      A.vmult(s, p);
      Number sigma = LinAlg::inner_product_tbb(s, p);
      // Step size in the correction direction.
      Number alpha = rho / sigma;
      // Update the solution vector.
      x.add(-alpha, p);
      // Update the residual vector.
      r.add(-alpha, s);
      // Update the preconditioned residual vector.
      preconditioner.vmult(v, r);
      // Update rho.
      rho = LinAlg::inner_product_tbb(v, r);

      if (control.log_history())
        {
          deallog << "step=" << i << ",residual_norm="
                  << numbers::NumberTraits<Number>::abs(rho) << std::endl;
        }

      if (numbers::NumberTraits<Number>::abs(rho) <=
          control.tolerance() * numbers::NumberTraits<Number>::abs(rho0))
        {
          // CG is convergent.
          return;
        }
      else
        {
          // Update the correction vector.
          Number beta = rho / rho_prev;
          p *= beta;
          p += v;
          rho_prev = rho;
        }
    }

  AssertThrow(
    false,
    SolverControl::NoConvergence(i, numbers::NumberTraits<Number>::abs(rho)));
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SOLVERS_SOLVER_CG_GENERAL_H_
