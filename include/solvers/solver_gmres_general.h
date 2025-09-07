/**
 * @file solver_gmres_general.h
 * @brief GMRES solver which can handle complex values.
 * @ingroup linalg
 *
 * @date 2025-07-24
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_SOLVERS_SOLVER_GMRES_GENERAL_H_
#define HIERBEM_INCLUDE_SOLVERS_SOLVER_GMRES_GENERAL_H_

#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_control.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>
#include <vector>

#include "config.h"
#include "linear_algebra/linalg.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * @brief Class for the direct version of GMRES with a restart mechanism.
 * @tparam VectorType
 */
template <typename VectorType>
class SolverDQGMRES
{
public:
  using size_type        = std::make_unsigned<types::blas_int>::type;
  using signed_size_type = types::blas_int;
  using VectorNumberType = typename VectorType::value_type;

  SolverDQGMRES(SolverControl &cn)
    : control(cn)
    , krylov_dim(30)
    , ortho_hist_len(krylov_dim)
    , is_left_precondition(true)
  {}

  ~SolverDQGMRES() = default;

  /**
   * Set the dimension of the maximum Krylov subspace.
   *
   * @param _krylov_dim Krylov subspace dimension
   */
  void
  set_krylov_dim(const size_type _krylov_dim)
  {
    krylov_dim = _krylov_dim;
  }

  /**
   * Set the number of of historic vectors used for orthogonalization during
   * constructing an orthonormal basis of the Krylov subspace.
   *
   * @param _ortho_hist_len Number of historic vectors. When it is equal to the
   * Krylov subspace dimension, full orthogonalization is adopted.
   */
  void
  set_orthogonal_history_length(const size_type _ortho_hist_len)
  {
    ortho_hist_len = _ortho_hist_len;
  }

  /**
   * Set if left or right precondition is used.
   */
  void
  set_left_precondition(const bool _is_left_precondition)
  {
    is_left_precondition = _is_left_precondition;
  }

  /**
   * Solve the matrix using DQGMRES.
   *
   * @param A System matrix to be solved
   * @param x Solution vector, its initial values will be used for computing the
   * initial residual vector. After execution of this function, @p x will be
   * updated.
   * @param b Right hand side vector
   * @param preconditioner Preconditioner object which should implement a @p vmult
   * method. N.B. When @p PreconditionJacobi is adopted, the class used for the
   * system matrix should implement a method @p precondition_jacobi .
   */
  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType         &A,
        VectorType               &x,
        const VectorType         &b,
        const PreconditionerType &preconditioner);

private:
  SolverControl &control;

  /**
   * @brief Krylov subspace dimension.
   */
  size_type krylov_dim;

  /**
   * @brief Number of historical vectors used in orthogonalization.
   */
  size_type ortho_hist_len;

  /**
   * @brief Whether the preconditioner is applied to the left or right.
   */
  bool is_left_precondition;
};


template <typename VectorType>
template <typename MatrixType, typename PreconditionerType>
void
SolverDQGMRES<VectorType>::solve(const MatrixType         &A,
                                 VectorType               &x,
                                 const VectorType         &b,
                                 const PreconditionerType &preconditioner)
{
  using real_type = typename numbers::NumberTraits<VectorNumberType>::real_type;

  // At the moment, @p LAPACKFullMatrixExt or @p dealii::SparseMatrix uses
  // @p m() and @p n() to get matrix sizes while @p HMaxtrix uses @p get_m()
  // and @p get_n(). Because they are not consistent, we do not perform a
  // dimension assertion about the matrix here.
  AssertDimension(x.size(), b.size());
  LogStream::Prefix prefix("dqgmres");

  // Vector size.
  const size_type n = b.size();

  // Orthonormal basis vectors of the Krylov subspace and an additional vector
  // which is orthogonal to the Krylov subspace. The solution vector is updated
  // as \f$x_m = x_0 + V_m y_m\f$.
  std::vector<VectorType> Vm(krylov_dim + 1);
  for (size_type i = 0; i < krylov_dim + 1; i++)
    Vm[i].reinit(n);

  // The last column of the Hessenberg matrix \f$\overline{H}_m\f$. In the inner
  // loop of this GMRES method, the Krylov subspace dimension incrementally
  // increases from 1 to \f$m\f$, so the size of the last column of
  // \f$\overline{H}_m\f$ should also change in principle. However, to reduce
  // repeated memory allocation and release, we directly allocate a vector with
  // the maximum size @p krylov_dim+1.
  VectorType hm(krylov_dim + 1);

  // Vectors which are transformed from the orthonormal basis vectors of the
  // Krylov subspace, i.e. \f$P_m = V_m R_m^{-1}\f$. Then the solution vector is
  // updated as \f$x_m = x_0 + P_m g_m\f$.
  std::vector<VectorType> Pm(krylov_dim);
  for (size_type i = 0; i < krylov_dim; i++)
    Pm[i].reinit(n);

  // Assistant vector for the error estimate when <code>ortho_hist_len <
  // krylov_dim</code>, i.e. incomplete Gram-Schmidt orthogonalization is
  // adopted. Therefore, we only accloate effective memory for @p Zm under this
  // condition.
  VectorType Zm;
  if (ortho_hist_len < krylov_dim)
    Zm.reinit(n);

  // Givens transformation matrices used for transforming the Hessenberg matrix
  // \f$\overline{H}_m\f$ into the upper triangular matrix \f$\overline{R}_m\f$.
  std::vector<FullMatrix<VectorNumberType>> givens_mats(krylov_dim);
  for (size_type i = 0; i < krylov_dim; i++)
    givens_mats[i].reinit(2, 2);

  // Initial residual vector.
  VectorType r0(n);

  // Residual norm at each step.
  real_type r_norm = 0.;

  // Temporary vector used for applying the preconditioner.
  VectorType v_precond(n);

  // The last entry of the right hand side vector \f$\overline{g}_m\f$ of size
  // \f$m+1\f$ in the least square problem, which has been applied \f$m\f$-times
  // of the Givens transformation.
  VectorNumberType gamma_next;
  // The last entry of the right hand side vector \f$g_m\f$ of size \f$m\f$ in
  // the least square problem. \f$g_m\f$ is the first \f$m\f$ entries of
  // \f$\overline{g}_m\f$.
  VectorNumberType gamma_current;

  // Outer loop for restarting the GMRES.
  unsigned int restart_index = 1;
  for (; restart_index <= control.max_steps(); restart_index++)
    {
      // Compute the initial residual vector.
      if (is_left_precondition)
        {
          A.vmult(v_precond, x);
          v_precond.sadd(VectorNumberType(-1.0), b);
          preconditioner.vmult(r0, v_precond);
        }
      else
        {
          A.vmult(r0, x);
          r0.sadd(VectorNumberType(-1.0), b);
        }

      r_norm        = r0.l2_norm();
      gamma_current = VectorNumberType(r_norm);
      if (restart_index == 1)
        {
          if (control.log_result())
            deallog << "restart step=0,residual norm=" << r_norm << std::endl;

          if (r_norm < control.tolerance())
            return;
        }

      // Clear the matrices and vectors.
      for (size_type i = 0; i < krylov_dim + 1; i++)
        Vm[i].reinit(n);

      for (size_type i = 0; i < krylov_dim; i++)
        {
          Pm[i].reinit(n);
          givens_mats[i].reinit(2, 2);
        }

      // Compute the first normalized basis vector for the Krylov subspace.
      Vm[0] = r0;
      Vm[0] /= gamma_current;

      // When incomplete Gram-Schmidt orthogonalization is adopted, we also
      // need to initialize the assistant vector @p Zm, which is used for an
      // accurate error estimate.
      if (ortho_hist_len < krylov_dim)
        Zm = Vm[0];

      // Inner loop for increasing the Krylov subspace dimension from 1 to @p m.
      for (size_type j = 0; j < krylov_dim; j++)
        {
          // Perform a full or incomplete Gram-Schmidt orthogonalization.
          if (is_left_precondition)
            {
              A.vmult(v_precond, Vm[j]);
              preconditioner.vmult(Vm[j + 1], v_precond);
            }
          else
            {
              preconditioner.vmult(v_precond, Vm[j]);
              A.vmult(Vm[j + 1], v_precond);
            }

          // Compute the last column of the Hessenberg matrix
          // \f$\overline{H}_j\f$.
          hm.reinit(krylov_dim + 1);
          for (size_type i =
                 std::max(0,
                          static_cast<signed_size_type>(j) -
                            static_cast<signed_size_type>(ortho_hist_len) + 1);
               i <= j;
               i++)
            {
              hm(i) = LinAlg::inner_product_tbb(Vm[j + 1], Vm[i]);
              Vm[j + 1].add(-hm(i), Vm[i]);
            }

          hm(j + 1) = VectorNumberType(Vm[j + 1].l2_norm());
          Vm[j + 1] /= hm(j + 1);

          // Apply historic Givens transformation to the last column of
          // \f$\overline{H}_j\f$.
          for (size_type i =
                 std::max(0,
                          static_cast<signed_size_type>(j) -
                            static_cast<signed_size_type>(ortho_hist_len));
               static_cast<signed_size_type>(i) <=
               static_cast<signed_size_type>(j) - 1;
               i++)
            LinAlg::apply_givens_2x2(hm, givens_mats[i], i);

          // Compute the current Givens transformation matrix.
          real_type denominator = 0.;
          if constexpr (numbers::NumberTraits<VectorNumberType>::is_complex)
            denominator = std::sqrt(std::abs(hm(j)) * std::abs(hm(j)) +
                                    std::abs(hm(j + 1)) * std::abs(hm(j + 1)));
          else
            denominator = std::sqrt(hm(j) * hm(j) + hm(j + 1) * hm(j + 1));

          VectorNumberType sj = hm(j + 1) / denominator;
          VectorNumberType cj = hm(j) / denominator;

          if constexpr (numbers::NumberTraits<VectorNumberType>::is_complex)
            {
              givens_mats[j](0, 0) = std::conj(cj);
              givens_mats[j](0, 1) = std::conj(sj);
              givens_mats[j](1, 0) = -sj;
              givens_mats[j](1, 1) = cj;
            }
          else
            {
              givens_mats[j](0, 0) = cj;
              givens_mats[j](0, 1) = sj;
              givens_mats[j](1, 0) = -sj;
              givens_mats[j](1, 1) = cj;
            }

          // Apply the current Givens transformation to the last column of the
          // Hessenberg matrix and the right hand side vector.
          hm(j) =
            givens_mats[j](0, 0) * hm(j) + givens_mats[j](0, 1) * hm(j + 1);
          hm(j + 1) = VectorNumberType(0.);

          gamma_next    = givens_mats[j](1, 0) * gamma_current;
          gamma_current = givens_mats[j](0, 0) * gamma_current;

          // Compute the vector @p pj.
          Pm[j] = Vm[j];
          for (size_type i =
                 std::max(0,
                          static_cast<signed_size_type>(j) -
                            static_cast<signed_size_type>(ortho_hist_len));
               static_cast<signed_size_type>(i) <=
               static_cast<signed_size_type>(j) - 1;
               i++)
            Pm[j].add(-hm(i), Pm[i]);

          Pm[j] /= hm(j);

          // Update the solution vector @p x.
          if (is_left_precondition)
            x.add(gamma_current, Pm[j]);
          else
            {
              preconditioner.vmult(v_precond, Pm[j]);
              x.add(gamma_current, v_precond);
            }

          // Compute the residual norm.
          if (ortho_hist_len < krylov_dim)
            {
              // Incomplete orthogonalization is adopted.
              Zm.sadd(givens_mats[j](1, 0), givens_mats[j](1, 1), Vm[j + 1]);
              r_norm = std::abs(gamma_next) * Zm.l2_norm();
            }
          else
            {
              // Full orthogonalization is adopted.
              r_norm = std::abs(gamma_next);
            }

          if (control.log_result())
            deallog << "restart step=" << restart_index
                    << ",inner iteration=" << j << ",residual norm=" << r_norm
                    << std::endl;

          // Check convergence.
          if (r_norm <= control.tolerance())
            return;
          else
            gamma_current = gamma_next;
        }
    }

  AssertThrow(false, SolverControl::NoConvergence(restart_index, r_norm));
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_SOLVERS_SOLVER_GMRES_GENERAL_H_
