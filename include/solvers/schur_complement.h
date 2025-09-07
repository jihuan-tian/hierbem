#ifndef HIERBEM_INCLUDE_SOLVERS_SCHUR_COMPLEMENT_H_
#define HIERBEM_INCLUDE_SOLVERS_SCHUR_COMPLEMENT_H_

#include <deal.II/lac/solver_control.h>

#include "linear_algebra/blas_helpers.h"
#include "solvers/solver_cg_general.h"

HBEM_NS_OPEN

/**
 * Class for the Schur complement with respet to the skew symmetric block
 * matrix.
 * TODO: enter the expression here.
 */
template <typename SkewSymmetricBlockMatrixType,
          typename VectorType,
          typename PrecondM11Type,
          typename Number>
class SchurComplement
{
public:
  SchurComplement(const SkewSymmetricBlockMatrixType &block_matrix_,
                  const PrecondM11Type               &precond11_,
                  const unsigned int                  max_iter_,
                  const Number                        tolerance_,
                  const bool                          log_history_,
                  const bool                          log_result_)
    : block_matrix(block_matrix_)
    , precond11(precond11_)
    , max_iter(max_iter_)
    , tolerance(tolerance_)
    , log_history(log_history_)
    , log_result(log_result_)
  {
    y1 = new VectorType();
    y2 = new VectorType();
    y3 = new VectorType();

    reinit();
  }

  ~SchurComplement()
  {
    delete y1;
    delete y2;
    delete y3;
  }

  void
  reinit();

  void
  vmult(VectorType &y, const VectorType &x) const;

  template <typename Number2>
  void
  vmult_add(VectorType &y, const Number2 alpha, const VectorType &x) const;

private:
  const SkewSymmetricBlockMatrixType &block_matrix;
  const PrecondM11Type               &precond11;
  unsigned int                        max_iter;
  Number                              tolerance;
  bool                                log_history;
  bool                                log_result;

  /**
   * Result vector of \f$M_{12}x_2\f$.
   */
  VectorType *y1;
  /**
   * Result vector of \f$M_{11}^{-1} M_12 x_2\f$.
   */
  VectorType *y2;
  /**
   * Result vector of \f$M_{12}^{\mathrm{T}} M_{11}^{-1} M_12 x_2\f$.
   */
  VectorType *y3;
};


template <typename SkewSymmetricBlockMatrixType,
          typename VectorType,
          typename PrecondM11Type,
          typename Number>
void
SchurComplement<SkewSymmetricBlockMatrixType,
                VectorType,
                PrecondM11Type,
                Number>::reinit()
{
  y1->reinit(block_matrix.get_M12().get_m());
  y2->reinit(block_matrix.get_M11().get_n());
  y3->reinit(block_matrix.get_M12().get_n());
}


template <typename SkewSymmetricBlockMatrixType,
          typename VectorType,
          typename PrecondM11Type,
          typename Number>
void
SchurComplement<SkewSymmetricBlockMatrixType,
                VectorType,
                PrecondM11Type,
                Number>::vmult(VectorType &y, const VectorType &x) const
{
  // It is possible that M12 is a non-symmetric H-matrix, whose vmult function
  // is accumulative. Therefore, we set the result vector as zero first to clear
  // previous values.
  block_matrix.get_M12().vmult(*y1, x);
  SolverControl solver_control(max_iter, tolerance, log_history, log_result);
  SolverCGGeneral<VectorType> solver(solver_control);
  solver.solve(block_matrix.get_M11(), *y2, *y1, precond11);
  block_matrix.get_M12().Tvmult(*y3, *y2);
  block_matrix.get_M22().vmult(y, x);
  y += *y3;
}


template <typename SkewSymmetricBlockMatrixType,
          typename VectorType,
          typename PrecondM11Type,
          typename Number>
template <typename Number2>
void
SchurComplement<SkewSymmetricBlockMatrixType,
                VectorType,
                PrecondM11Type,
                Number>::vmult_add(VectorType       &y,
                                   const Number2     alpha,
                                   const VectorType &x) const
{
  VectorType y_tmp(y.size());
  vmult(y_tmp, x);
  BLASHelpers::scal_helper(y_tmp.size(),
                           typename VectorType::value_type(alpha),
                           y_tmp);
  y += y_tmp;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SOLVERS_SCHUR_COMPLEMENT_H_
