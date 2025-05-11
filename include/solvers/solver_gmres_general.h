/**
 * @file solver_gmres_general.h
 * @brief GMRES solver which can handle complex values.
 * @ingroup linalg
 *
 * @date 2025-05-10
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_SOLVERS_SOLVER_GMRES_GENERAL_H_
#define HIERBEM_INCLUDE_SOLVERS_SOLVER_GMRES_GENERAL_H_

#include <deal.II/lac/solver_control.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

template <typename VectorType>
class SolverGMRESGeneral
{
public:
  using size_type = types::blas_int;
  using Number    = typename VectorType::value_type;

  SolverGMRESGeneral(SolverControl &cn)
    : control(cn)
  {}

  ~SolverGMRESGeneral() = default;

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
SolverGMRESGeneral<VectorType>::solve(const MatrixType         &A,
                                      VectorType               &x,
                                      const VectorType         &b,
                                      const PreconditionerType &preconditioner)
{
  (void)A;
  (void)x;
  (void)b;
  (void)preconditioner;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SOLVERS_SOLVER_GMRES_GENERAL_H_
