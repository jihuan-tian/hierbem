/**
 * \file rkmatrix.h
 * \brief Definition of rank-k matrix.
 * \ingroup hierarchical_matrices
 * \date 2021-06-06
 * \author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_HMATRIX_RKMATRIX_H_
#define HIERBEM_INCLUDE_HMATRIX_RKMATRIX_H_

#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/lapack_templates.h>

#include <cmath>
#include <complex>
#include <map>
#include <type_traits>
#include <vector>

#include "config.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

HBEM_NS_OPEN

using namespace dealii;

template <int spacedim, typename Number>
class HMatrix;

template <typename Number = double>
class RkMatrix
{
public:
  /**
   * Declare the type for container size.
   */
  using size_type  = std::make_unsigned<types::blas_int>::type;
  using value_type = Number;
  using real_type  = typename numbers::NumberTraits<Number>::real_type;

  // Friend functions for \f$\mathcal{H}\f$-matrix arithmetic operations.
  template <int spacedim1, typename Number1>
  friend void
  h_rk_mmult(HMatrix<spacedim1, Number1> &M1,
             const RkMatrix<Number1>     &M2,
             RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_mmult(const Number1                alpha,
             HMatrix<spacedim1, Number1> &M1,
             const RkMatrix<Number1>     &M2,
             RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_mTmult(HMatrix<spacedim1, Number1> &M1,
              const RkMatrix<Number1>     &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_mTmult(const Number1                alpha,
              HMatrix<spacedim1, Number1> &M1,
              const RkMatrix<Number1>     &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_Tmmult(HMatrix<spacedim1, Number1> &M1,
              const RkMatrix<Number1>     &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_Tmmult(const Number1                alpha,
              HMatrix<spacedim1, Number1> &M1,
              const RkMatrix<Number1>     &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  h_rk_mmult_for_h_h_mmult(HMatrix<spacedim1, Number1>       *M1,
                           const HMatrix<spacedim1, Number1> *M2,
                           HMatrix<spacedim1, Number1>       *M,
                           bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mmult(const RkMatrix<Number1>     &M1,
             HMatrix<spacedim1, Number1> &M2,
             RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mmult(const Number1                alpha,
             const RkMatrix<Number1>     &M1,
             HMatrix<spacedim1, Number1> &M2,
             RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mTmult(const RkMatrix<Number1>     &M1,
              HMatrix<spacedim1, Number1> &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mTmult(const Number1                alpha,
              const RkMatrix<Number1>     &M1,
              HMatrix<spacedim1, Number1> &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_Tmmult(const RkMatrix<Number1>     &M1,
              HMatrix<spacedim1, Number1> &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_Tmmult(const Number1                alpha,
              const RkMatrix<Number1>     &M1,
              HMatrix<spacedim1, Number1> &M2,
              RkMatrix<Number1>           &M);

  template <int spacedim1, typename Number1>
  friend void
  rk_h_mmult_for_h_h_mmult(const HMatrix<spacedim1, Number1> *M1,
                           HMatrix<spacedim1, Number1>       *M2,
                           HMatrix<spacedim1, Number1>       *M,
                           bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult(HMatrix<spacedim1, Number1>        &M1,
            const LAPACKFullMatrixExt<Number1> &M2,
            LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult(const Number1                       alpha,
            HMatrix<spacedim1, Number1>        &M1,
            const LAPACKFullMatrixExt<Number1> &M2,
            LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mTmult(HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mTmult(const Number1                       alpha,
             HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_Tmmult(HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_Tmmult(const Number1                       alpha,
             HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult(HMatrix<spacedim1, Number1>        &M1,
            const LAPACKFullMatrixExt<Number1> &M2,
            RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult(const Number1                       alpha,
            HMatrix<spacedim1, Number1>        &M1,
            const LAPACKFullMatrixExt<Number1> &M2,
            RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mTmult(HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mTmult(const Number1                       alpha,
             HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_Tmmult(HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_Tmmult(const Number1                       alpha,
             HMatrix<spacedim1, Number1>        &M1,
             const LAPACKFullMatrixExt<Number1> &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  h_f_mmult_for_h_h_mmult(HMatrix<spacedim1, Number1>       *M1,
                          const HMatrix<spacedim1, Number1> *M2,
                          HMatrix<spacedim1, Number1>       *M,
                          bool is_M1M2_last_in_M_Sigma_P);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult(const LAPACKFullMatrixExt<Number1> &M1,
            HMatrix<spacedim1, Number1>        &M2,
            LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult(const Number1                       alpha,
            const LAPACKFullMatrixExt<Number1> &M1,
            HMatrix<spacedim1, Number1>        &M2,
            LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mTmult(const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mTmult(const Number1                       alpha,
             const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_Tmmult(const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_Tmmult(const Number1                       alpha,
             const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             LAPACKFullMatrixExt<Number1>       &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult(const LAPACKFullMatrixExt<Number1> &M1,
            HMatrix<spacedim1, Number1>        &M2,
            RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult(const Number1                       alpha,
            const LAPACKFullMatrixExt<Number1> &M1,
            HMatrix<spacedim1, Number1>        &M2,
            RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mTmult(const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mTmult(const Number1                       alpha,
             const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_Tmmult(const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_Tmmult(const Number1                       alpha,
             const LAPACKFullMatrixExt<Number1> &M1,
             HMatrix<spacedim1, Number1>        &M2,
             RkMatrix<Number1>                  &M);

  template <int spacedim1, typename Number1>
  friend void
  f_h_mmult_for_h_h_mmult(const HMatrix<spacedim1, Number1> *M1,
                          HMatrix<spacedim1, Number1>       *M2,
                          HMatrix<spacedim1, Number1>       *M,
                          bool is_M1M2_last_in_M_Sigma_P);

  template <typename Number1>
  friend void
  print_rkmatrix_to_mat(std::ostream            &out,
                        const std::string       &name,
                        const RkMatrix<Number1> &values,
                        const unsigned int       precision,
                        const bool               scientific,
                        const unsigned int       width,
                        const char              *zero_string,
                        const double             denominator,
                        const double             threshold);

  /**
   * Default constructor.
   */
  RkMatrix();

  /**
   * Construct a zero-valued rank-k matrix with the specified matrix dimension
   *
   * and rank.
   * @param m
   * @param n
   * @param fixed_rank_k
   */
  RkMatrix(const size_type m, const size_type n, const size_type fixed_rank_k);

  /**
   * Construct a rank-k matrix by conversion from a full matrix \p M with rank
   * truncation.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method converts a full matrix to a rank-k matrix, which
   * implements the operator \f$\mathcal{T}_{r}^{\mathcal{R} \leftarrow
   * \mathcal{F}}\f$ in (7.2) in Hackbusch's \f$\mathcal{H}\f$-matrix book.
   * The
   * original full matrix \p will be modified since SVD will be applied to
   * it.</dd>
   * </dl>
   * @param fixed_rank_k
   * @param M
   */
  RkMatrix(const size_type fixed_rank_k, LAPACKFullMatrixExt<Number> &M);

  /**
   * Construct a rank-k matrix by conversion from a full matrix \p M with rank
   * truncation. The error matrices are returned.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method converts a full matrix to a rank-k matrix, which
   * implements the operator \f$\mathcal{T}_{r}^{\mathcal{R} \leftarrow
   * \mathcal{F}}\f$ in (7.2) in Hackbusch's \f$\mathcal{H}\f$-matrix book.
   * The
   * original full matrix \p will be modified since SVD will be applied to
   * it.</dd>
   * </dl>
   * @param fixed_rank_k
   * @param M
   * @param C
   * @param D
   */
  RkMatrix(const size_type              fixed_rank_k,
           LAPACKFullMatrixExt<Number> &M,
           LAPACKFullMatrixExt<Number> &C,
           LAPACKFullMatrixExt<Number> &D);

  /**
   * Construct a rank-k matrix by conversion from a full matrix \p M without
   * rank truncation.
   *
   * \mynote{The original full matrix \p will be modified since SVD will be
   * applied to it.}
   *
   * @param M
   */
  RkMatrix(LAPACKFullMatrixExt<Number> &M);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ with rank truncation from the **full global** matrix @p M
   * defined on the complete block cluster \f$I \times J\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This operation will not modify the full global matrix \p M.</dd>
   * </dl>
   * @param tau
   * @param sigma
   * @param fixed_rank_k
   * @param M
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const size_type                               fixed_rank_k,
           const LAPACKFullMatrixExt<Number>            &M);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ without rank truncation from the **full global** matrix
   * @p M defined on the complete block cluster \f$I \times J\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This operation will not modify the full global matrix \p M.</dd>
   * </dl>
   * @param tau
   * @param sigma
   * @param M
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const LAPACKFullMatrixExt<Number>            &M);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ with rank truncation from the **full local* matrix @p M.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The original full local matrix \p M will not be modified.</dd>
   * </dl>
   *
   * @param tau_index_range
   * @param sigma_index_range
   * @param fixed_rank_k
   * @param M
   * @param M_tau_index_range
   * @param M_sigma_index_range
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const size_type                               fixed_rank_k,
           const LAPACKFullMatrixExt<Number>            &M,
           const std::array<types::global_dof_index, 2> &M_tau_index_range,
           const std::array<types::global_dof_index, 2> &M_sigma_index_range);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ without rank truncation from the **full local** matrix
   * @p M.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The original full local matrix \p M will not be modified.</dd>
   * </dl>
   *
   * @param tau_index_range
   * @param sigma_index_range
   * @param M
   * @param M_tau_index_range
   * @param M_sigma_index_range
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const LAPACKFullMatrixExt<Number>            &M,
           const std::array<types::global_dof_index, 2> &M_tau_index_range,
           const std::array<types::global_dof_index, 2> &M_sigma_index_range);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ with rank truncation from the **global rank-k** matrix
   * @p M.
   *
   * For a rank-k matrix \f$M=A\cdot B^T\f$, it is actually the sum of a
   * series of outer products which are formed by the rows of \f$A\f$ and
   * \f$B\f$, i.e. \f$M(i,j)=\sum_{i=1}^k A(i,:) B(j,:)^T\f$. Then restricting
   * a rank-k matrix to a block cluster \f$\tau\times\sigma\f$ is actually
   * computed as \f$A(\tau,:)\cdot B(\sigma,:)^T\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The original rank-k global matrix \p M will not be modified.</dd>
   * </dl>
   * @param tau
   * @param sigma
   * @param fixed_rank_k
   * @param M
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const size_type                               fixed_rank_k,
           const RkMatrix<Number>                       &M);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ without rank truncation from the **global rank-k**
   * matrix
   * @p M.
   *
   * For a rank-k matrix \f$M=A\cdot B^T\f$, it is actually the sum of a
   * series of outer products which are formed by the rows of \f$A\f$ and
   * \f$B\f$, i.e. \f$M(i,j)=\sum_{i=1}^k A(i,:) B(j,:)^T\f$. Then restricting
   * a rank-k matrix to a block cluster \f$\tau\times\sigma\f$ is actually
   * computed as \f$A(\tau,:)\cdot B(\sigma,:)^T\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The original rank-k global matrix \p M will not be modified.</dd>
   * </dl>
   * @param tau
   * @param sigma
   * @param M
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const RkMatrix<Number>                       &M);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ with rank truncation from the **local rank-k** matrix
   * @p M.
   *
   * For a rank-k matrix \f$M=A\cdot B^T\f$, it is actually the sum of a
   * series of outer products which are formed by the rows of \f$A\f$ and
   * \f$B\f$, i.e. \f$M(i,j)=\sum_{i=1}^k A(i,:) B(j,:)^T\f$. Then restricting
   * a rank-k matrix to a block cluster \f$\tau\times\sigma\f$ is actually
   * computed as \f$A(\tau,:)\cdot B(\sigma,:)^T\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The original rank-k local matrix \p M will not be modified.</dd>
   * </dl>
   *
   * @param tau_index_range
   * @param sigma_index_range
   * @param fixed_rank_k
   * @param M
   * @param M_tau_index_range
   * @param M_sigma_index_range
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const size_type                               fixed_rank_k,
           const RkMatrix<Number>                       &M,
           const std::array<types::global_dof_index, 2> &M_tau_index_range,
           const std::array<types::global_dof_index, 2> &M_sigma_index_range);

  /**
   * Construct a rank-k matrix by restriction to the block cluster \f$\tau
   * \times \sigma\f$ without rank truncation from the **local rank-k** matrix
   * @p M. The rank of the rank-k matrix to be constructed is initialized to be
   * the minimum of its minimum matrix dimension and the rank of M.
   *
   * For a rank-k matrix \f$M=A\cdot B^T\f$, it is actually the sum of a
   * series of outer products which are formed by the rows of \f$A\f$ and
   * \f$B\f$, i.e. \f$M(i,j)=\sum_{i=1}^k A(i,:) B(j,:)^T\f$. Then restricting
   * a rank-k matrix to a block cluster \f$\tau\times\sigma\f$ is actually
   * computed as \f$A(\tau,:)\cdot B(\sigma,:)^T\f$.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The original rank-k matrix \p M will not be modified.</dd>
   * </dl>
   */
  RkMatrix(const std::array<types::global_dof_index, 2> &tau_index_range,
           const std::array<types::global_dof_index, 2> &sigma_index_range,
           const RkMatrix<Number>                       &M,
           const std::array<types::global_dof_index, 2> &M_tau_index_range,
           const std::array<types::global_dof_index, 2> &M_sigma_index_range);

  /**
   * Construct a rank-k matrix from two component matrices.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The formal rank of the rank-k matrix is set to the number of
   * columns
   * of matrix \p A or \p B. The rank of the rank-k matrix will not be
   * calculated but temporarily set to the minimum dimension of the matrix.
   * Hence we have \f$\text{actual rank} \leq \text{rank} \leq \text{formal
   * rank}\f$.</dd>
   * </dl>
   * @param A
   * @param B
   */
  RkMatrix(const LAPACKFullMatrixExt<Number> &A,
           const LAPACKFullMatrixExt<Number> &B);

  /**
   * Construct a rank-k matrix \f$M\f$ from an agglomeration of two rank-k
   * submatrices, \f$M_1\f$ and \f$M_2\f$, which have been obtained from
   * either horizontal splitting or vertical splitting.
   *
   * @param fixed_rank_k
   * @param M1
   * @param M2
   */
  RkMatrix(const size_type         fixed_rank_k,
           const RkMatrix<Number> &M1,
           const RkMatrix<Number> &M2,
           bool                    is_horizontal_split);

  /**
   * Construct a rank-k matrix \f$M\f$ from an agglomeration of four
   * rank-k submatrices, \f$M_{11}, M_{12}, M_{21}, M_{22}\f$.
   *
   * \f[
   * M =
   * \begin{pmatrix}
   * M_{11} & M_{12} \\
   * M_{21} & M_{22}
   * \end{pmatrix}
   * \f]
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method implements the operator \f$\mathcal{T}_{r, {\rm
   * pairw}}^{\mathcal{R}}\f$ in (2.13) in Hackbusch's
   * \f$\mathcal{H}\f$-matrix book.</dd>
   * </dl>
   */
  RkMatrix(const size_type         fixed_rank_k,
           const RkMatrix<Number> &M11,
           const RkMatrix<Number> &M12,
           const RkMatrix<Number> &M21,
           const RkMatrix<Number> &M22,
           const Number            rank_factor = 1.0);

  /**
   * Copy constructor.
   */
  RkMatrix(const RkMatrix<Number> &matrix);

  /**
   * Assignment operator.
   * @param matrix
   * @return
   */
  RkMatrix<Number> &
  operator=(const RkMatrix<Number> &matrix);

  /**
   * Reinitialize a rank-k matrix with specified dimension and rank. By
   * default, all matrix entries are initialized to zero.
   * @param m
   * @param n
   * @param fixed_rank_k
   * @param omit_zeroing_entries
   */
  void
  reinit(const size_type m, const size_type n, const size_type fixed_rank_k);

  /**
   * Get the number of rows.
   * @return
   */
  size_type
  get_m() const;

  /**
   * Get the number of columns.
   * @return
   */
  size_type
  get_n() const;

  /**
   * Get the rank of the rank-k matrix.
   * @return
   */
  size_type
  get_rank() const;

  /**
   * Get the formal rank of the rank-k matrix.
   * @return
   */
  size_type
  get_formal_rank() const;

  /**
   * Get the reference to the component matrix \p A.
   * @return
   */
  LAPACKFullMatrixExt<Number> &
  get_A();

  /**
   * Get the reference to the component matrix \p A (const version).
   * @return
   */
  const LAPACKFullMatrixExt<Number> &
  get_A() const;

  /**
   * Get the reference to the component matrix \p B.
   * @return
   */
  LAPACKFullMatrixExt<Number> &
  get_B();

  /**
   * Get the reference to the component matrix \p B (const version).
   * @return
   */
  const LAPACKFullMatrixExt<Number> &
  get_B() const;

  /**
   * Convert an rank-k matrix to a full matrix.
   *
   * @param matrix
   * @return Return true when the full matrix is not zero-valued.
   */
  bool
  convertToFullMatrix(LAPACKFullMatrixExt<Number> &matrix) const;

  /**
   * Restrict a **global rank-k** matrix to a full matrix defined on the block
   * cluster \f$\tau \times \sigma\f$.
   */
  void
  restrictToFullMatrix(
    const std::array<types::global_dof_index, 2> &tau_index_range,
    const std::array<types::global_dof_index, 2> &sigma_index_range,
    LAPACKFullMatrixExt<Number>                  &matrix) const;

  /**
   * Restrict a **local rank-k** matrix to a full matrix defined on the block
   * cluster \f$\tau \times \sigma\f$.
   *
   * @param rk_tau_index_range
   * @param rk_sigma_index_range
   * @param matrix
   * @param tau_index_range
   * @param sigma_index_range
   */
  void
  restrictToFullMatrix(
    const std::array<types::global_dof_index, 2> &rk_tau_index_range,
    const std::array<types::global_dof_index, 2> &rk_sigma_index_range,
    LAPACKFullMatrixExt<Number>                  &M,
    const std::array<types::global_dof_index, 2> &M_tau_index_range,
    const std::array<types::global_dof_index, 2> &M_sigma_index_range) const;

  /**
   * Print a RkMatrix.
   * @param out
   * @param precision
   * @param scientific
   * @param width
   * @param zero_string
   * @param denominator
   * @param threshold
   */
  void
  print_formatted(std::ostream      &out,
                  const unsigned int precision   = 3,
                  const bool         scientific  = true,
                  const unsigned int width       = 0,
                  const char        *zero_string = "0",
                  const double       denominator = 1.,
                  const double       threshold   = 0.) const;

  /**
   * Print a RkMatrix into Octave mat format.
   * @param out
   * @param name
   * @param precision
   * @param scientific
   * @param width
   * @param zero_string
   * @param denominator
   * @param threshold
   */
  void
  print_formatted_to_mat(std::ostream      &out,
                         const std::string &name,
                         const unsigned int precision   = 8,
                         const bool         scientific  = true,
                         const unsigned int width       = 0,
                         const char        *zero_string = "0",
                         const double       denominator = 1.,
                         const double       threshold   = 0.) const;

  /**
   * Keep the first n columns in both two component matrices by resizing.
   * @pre
   * @post
   * @param n
   */
  void
  keep_first_n_columns(const size_type n);

  /**
   * Truncate the RkMatrix to \p new_rank.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method implements the operator \f$\mathcal{T}_{r \leftarrow
   * s}^{\mathcal{R}}\f$ in (7.4) in Hackbusch's \f$\mathcal{H}\f$-matrix
   * book. If the actual rank of the matrix is less than the specified \p
   * new_rank, i.e. rank deficient, the matrix will be truncated to its actual
   * rank.</dd>
   * </dl>
   * @param new_rank
   */
  void
  truncate_to_rank(size_type new_rank);

  /**
   * Truncate the RkMatrix to \p new_rank with the error matrices returned.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method implements the operator \f$\mathcal{T}_{r \leftarrow
   * s}^{\mathcal{R}}\f$ in (7.4) in Hackbusch's \f$\mathcal{H}\f$-matrix
   * book. If the actual rank of the matrix is less than the specified \p
   * new_rank, i.e. rank deficient, the matrix will be truncated to its actual
   * rank.</dd>
   * </dl>
   * @param new_rank
   */
  void
  truncate_to_rank(size_type                    new_rank,
                   LAPACKFullMatrixExt<Number> &C,
                   LAPACKFullMatrixExt<Number> &D);

  /**
   * Calculate the upper bound of the matrix rank, which is \f$\min\{{\rm
   * rank}(A), {\rm rank}(B), rank\}\f$.
   */
  void
  calc_rank_upper_bound();

  /**
   * Calculate matrix-vector multiplication.
   *
   * @param y
   * @param x
   * @param adding
   */
  void
  vmult(Vector<Number>       &y,
        const Vector<Number> &x,
        const bool            adding = false) const;

  /**
   * Calculate matrix-vector multiplication with the matrix applied complex
   * conjugation.
   */
  void
  Cvmult(Vector<Number>       &y,
         const Vector<Number> &x,
         const bool            adding = false) const;

  /**
   * Calculate transposed matrix-vector multiplication.
   *
   * @param y
   * @param x
   * @param adding
   */
  void
  Tvmult(Vector<Number>       &y,
         const Vector<Number> &x,
         const bool            adding = false) const;

  /**
   * Calculate Hermite transposed matrix-vector multiplication.
   */
  void
  Hvmult(Vector<Number>       &y,
         const Vector<Number> &x,
         const bool            adding = false) const;

  /**
   * Perform the addition of two rank-k matrices \f$M = M_1 + M_2\f$ by
   * juxtaposition without rank truncation, where \f$M_1\f$ is the current
   * matrix.
   *
   * Let \f$M_1 = A_1 B_1^T\f$ and \f$M_2 = A_2 B_2^T\f$. Their addition
   * without truncation is a juxtaposition of the components of \f$M_1\f$ and
   * \f$M_2\f$. Assume \f$M = AB^T\f$, then \f$A = [A_1 A_2]\f$ and \f$B =
   * [B_1 B_2]\f$. Its rank \f$r \leq r_1 + r_2\f$. Also note that the value
   * of the class
   * member \p M may be just formal, i.e. the actual matrix rank is smaller than
   * it.
   * @param M
   * @param M2
   * @param fixed_rank_k
   */
  void
  add(RkMatrix<Number> &M, const RkMatrix<Number> &M2) const;

  /**
   * Perform the addition of two rank-k matrices \f$M = M_1 + m_2 M_2\f$ with
   * a factor \f$m_2\f$ multiplied to \f$M_2\f$. There is no rank truncation.
   */
  void
  add(RkMatrix<Number> &M, const Number m2, const RkMatrix<Number> &M2) const;

  /**
   * Perform the addition of two rank-k matrices \f$M = M + M_1\f$ by
   * juxtaposition without rank truncation, where \f$M\f$ is the current
   * matrix.
   * @param M1
   */
  void
  add(const RkMatrix<Number> &M1);

  /**
   * Perform the addition of two rank-k matrices \f$M = M + m_1 M_1\f$ with a
   * factor \f$m_1\f$ multiplied to \f$M_1\f$. There is no rank truncation.
   * @param M1
   */
  void
  add(const Number m1, const RkMatrix<Number> &M1);

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M_1 +
   * M_2\f$.
   * The resulted rank-k matrix \p M will be truncated to the fixed rank \p
   * fixed_rank_k.
   */
  void
  add(RkMatrix<Number>       &M,
      const RkMatrix<Number> &M2,
      const size_type         fixed_rank_k) const;

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M_1 +
   * M_2\f$.
   * The resulted rank-k matrix \p M will be truncated to the fixed rank \p
   * fixed_rank_k. The truncation error is returned as the product \f$CD^T\f$.
   *
   * @param M
   * @param M2
   * @param fixed_rank_k
   * @param C
   * @param D
   */
  void
  add(RkMatrix<Number>            &M,
      const RkMatrix<Number>      &M2,
      const size_type              fixed_rank_k,
      LAPACKFullMatrixExt<Number> &C,
      LAPACKFullMatrixExt<Number> &D) const;

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M_1 + m_2
   * M_2\f$ with a factor \f$m_2\f$ multiplied to \f$M_2\f$. The resulted
   * rank-k
   * matrix \p M will be truncated to the fixed rank \p fixed_rank_k.
   */
  void
  add(RkMatrix<Number>       &M,
      const Number            m2,
      const RkMatrix<Number> &M2,
      const size_type         fixed_rank_k) const;

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M_1 + m_2
   * M_2\f$ with a factor \f$m_2\f$ multiplied to \f$M_2\f$. The resulted
   * rank-k
   * matrix \p M will be truncated to the fixed rank \p fixed_rank_k. The
   * truncation error is returned as the product \f$CD^T\f$.
   *
   * @param M
   * @param m2
   * @param M2
   * @param fixed_rank_k
   * @param C
   * @param D
   */
  void
  add(RkMatrix<Number>            &M,
      const Number                 m2,
      const RkMatrix<Number>      &M2,
      const size_type              fixed_rank_k,
      LAPACKFullMatrixExt<Number> &C,
      LAPACKFullMatrixExt<Number> &D) const;

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M + M_1\f$.
   * \p M_1 is added to the current matrix itself. The resulted rank-k matrix \p
   * M will be truncated to the fixed rank \p fixed_rank_k.
   */
  void
  add(const RkMatrix<Number> &M1, const size_type fixed_rank_k);

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M + M_1\f$.
   * \p M_1 is added to the current matrix itself. The resulted rank-k matrix \p
   * M will be truncated to the fixed rank \p fixed_rank_k. The truncation
   * error is returned as the product \f$CD^T\f$.
   *
   * @param M1
   * @param fixed_rank_k
   * @param C
   * @param D
   */
  void
  add(const RkMatrix<Number>      &M1,
      const size_type              fixed_rank_k,
      LAPACKFullMatrixExt<Number> &C,
      LAPACKFullMatrixExt<Number> &D);

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M + m_1
   * M_1\f$ with factor \f$m_1\f$ multiplied to \f$M_1\f$. \p M_1 is added to
   * the current matrix itself. The resulted rank-k matrix \p M will be
   * truncated to the fixed rank \p fixed_rank_k.
   */
  void
  add(const Number            m1,
      const RkMatrix<Number> &M1,
      const size_type         fixed_rank_k);

  /**
   * Perform the formatted addition of two rank-k matrices, \f$M = M + m_1
   * M_1\f$ with factor \f$m_1\f$ multiplied to \f$M_1\f$. \p M_1 is added to
   * the current matrix itself. The resulted rank-k matrix \p M will be
   * truncated to the fixed rank \p fixed_rank_k. The truncation
   * error is returned as the product \f$CD^T\f$.
   *
   * @param m1
   * @param M1
   * @param fixed_rank_k
   * @param C
   * @param D
   */
  void
  add(const Number                 m1,
      const RkMatrix<Number>      &M1,
      const size_type              fixed_rank_k,
      LAPACKFullMatrixExt<Number> &C,
      LAPACKFullMatrixExt<Number> &D);

  /**
   * Assemble the smaller rank-k matrix \p M into the current rank-k matrix
   * with respect to their index ranges.
   *
   * @param tau_index_range
   * @param sigma_index_range
   * @param M
   * @param M_tau_index_range
   * @param M_sigma_index_range
   * @param fixed_rank_k
   */
  void
  assemble_from_rkmatrix(
    const std::array<types::global_dof_index, 2> &tau_index_range,
    const std::array<types::global_dof_index, 2> &sigma_index_range,
    const RkMatrix<Number>                       &M,
    const std::array<types::global_dof_index, 2> &M_tau_index_range,
    const std::array<types::global_dof_index, 2> &M_sigma_index_range,
    const size_type                               fixed_rank_k);

  /**
   * Assemble the smaller rank-k matrix \p M into the current rank-k matrix with
   * respect to their index ranges The truncation error is returned as the
   * product \f$CD^T\f$.
   *
   * @param tau_index_range
   * @param sigma_index_range
   * @param M
   * @param M_tau_index_range
   * @param M_sigma_index_range
   * @param fixed_rank_k
   * @param C
   * @param D
   */
  void
  assemble_from_rkmatrix(
    const std::array<types::global_dof_index, 2> &tau_index_range,
    const std::array<types::global_dof_index, 2> &sigma_index_range,
    const RkMatrix<Number>                       &M,
    const std::array<types::global_dof_index, 2> &M_tau_index_range,
    const std::array<types::global_dof_index, 2> &M_sigma_index_range,
    const size_type                               fixed_rank_k,
    LAPACKFullMatrixExt<Number>                  &C,
    LAPACKFullMatrixExt<Number>                  &D);

  /**
   * Calculate the Frobenius norm of the rank-k matrix.
   *
   * @param number_of_columns The effective number of columns in \f$A\f$ and
   * \f$B\f$ to be included into the calculation.
   *
   * @return
   */
  real_type
  frobenius_norm(const size_type number_of_columns) const;

  /**
   * Calculate the Frobenius norm of the rank-k matrix using all the columns
   * in the component matrices \f$A\f$ and \f$B\f$.
   *
   * @return
   */
  real_type
  frobenius_norm(void) const;

  /**
   * Determine an estimate for the memory consumption (in bytes) of this
   * object.
   *
   * @return
   */
  std::size_t
  memory_consumption() const;

private:
  LAPACKFullMatrixExt<Number> A;
  LAPACKFullMatrixExt<Number> B;

  /**
   * Matrix rank, which is either the actual matrix rank or the minimum of \p m,
   * \p n and \p A.n (or \p B.n). Actually, this is an upper bound of the actual
   * matrix rank.
   */
  size_type rank;

  /**
   * Formal matrix rank, which is equal to the number of columns of \p A or \p
   * B, i.e. \p A.n or \p B.n.
   */
  size_type formal_rank;

  /**
   * Total number of rows.
   */
  size_type m;

  /**
   * Total number of columns
   */
  size_type n;
};


template <typename Number>
RkMatrix<Number>::RkMatrix()
  : A(0, 0)
  , B(0, 0)
  , rank(0)
  , formal_rank(0)
  , m(0)
  , n(0)
{}


template <typename Number>
RkMatrix<Number>::RkMatrix(const size_type m,
                           const size_type n,
                           const size_type fixed_rank_k)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(m)
  , n(n)
{
  /**
   * If the given \p fixed_rank_k is larger than the minimum matrix dimension
   * \f$\min\{m, n\}\f$, simply set it as this value.
   */
  const size_type min_dim        = std::min(m, n);
  const size_type effective_rank = std::min(min_dim, fixed_rank_k);

  //  // DEBUG: check the consistency between the specified rank and the
  //  effective
  //  // rank.
  //  std::cout << "min_dim=" << min_dim << ", effective_rank=" <<
  //  effective_rank
  //            << ", fixed_rank_k=" << fixed_rank_k << std::endl;

  if (effective_rank > 0 && m != 0 && n != 0)
    {
      /**
       * \alert{Only when the effective rank is larger than zero and the
       * matrix has non-zero row or column dimension, memory for the component
       * matrices is allocated.}
       */
      A.reinit(m, effective_rank);
      B.reinit(n, effective_rank);
      /**
       * \alert{Even though the component matrices are zero-valued at the
       * moment, because the current rank-k matrix is to be assigned with
       * effect
       * data, its rank should be still set to the \p effective_rank.}
       */
      rank        = effective_rank;
      formal_rank = effective_rank;
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const size_type              fixed_rank_k,
                           LAPACKFullMatrixExt<Number> &M)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(M.m())
  , n(M.n())
{
  if (m == 0 || n == 0 || fixed_rank_k == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero or
       * the given rank is zero, do not allocate memory for the component
       * matrices of the rank-k matrix, since there is no effective data.}
       */
    }
  else
    {
      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M(0, 0)) ? -1.0 : 1.0);
                }
            }
        }
      else
        {
          rank        = M.rank_k_decompose(fixed_rank_k, A, B);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const size_type              fixed_rank_k,
                           LAPACKFullMatrixExt<Number> &M,
                           LAPACKFullMatrixExt<Number> &C,
                           LAPACKFullMatrixExt<Number> &D)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(M.m())
  , n(M.n())
{
  if (m == 0 || n == 0 || fixed_rank_k == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero or
       * the given rank is zero, do not allocate memory for the component
       * matrices of the rank-k matrix, since there is no effective data.}
       */

      /**
       * Set the error matrices as zero.
       */
      C.reinit(0, 0);
      D.reinit(0, 0);
    }
  else
    {
      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M(0, 0)) ? -1.0 : 1.0);
                }
            }

          /**
           * Set the error matrices as zero.
           */
          C.reinit(0, 0);
          D.reinit(0, 0);
        }
      else
        {
          rank        = M.rank_k_decompose(fixed_rank_k, A, B, C, D);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(LAPACKFullMatrixExt<Number> &M)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(M.m())
  , n(M.n())
{
  if (m == 0 || n == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero, do
       * not allocate memory for the component matrices of the rank-k matrix,
       * since there is no effective data.}
       */
    }
  else
    {
      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M(0, 0)) ? -1.0 : 1.0);
                }
            }
        }
      else
        {
          rank        = M.rank_k_decompose(A, B);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const size_type                               fixed_rank_k,
  const LAPACKFullMatrixExt<Number>            &M)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  /**
   * Convert the matrix block \p M_b in full matrix format to rank-k matrix
   * format.
   */
  if (m == 0 || n == 0 || fixed_rank_k == 0 || M.m() == 0 || M.n() == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero or
       * the given rank is zero, do not allocate memory for the component
       * matrices of the rank-k matrix, since there is no effective data.}
       */
    }
  else
    {
      /**
       * Extract the data for the submatrix defined on the block cluster
       * \f$\tau
       * \times \sigma\f$ from the full global matrix \p M.
       */
      LAPACKFullMatrixExt<Number> M_b(tau_index_range, sigma_index_range, M);

      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M_b(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M_b(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M_b(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M_b(0, 0)) ? -1.0 : 1.0);
                }
            }
        }
      else
        {
          rank        = M_b.rank_k_decompose(fixed_rank_k, A, B);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const LAPACKFullMatrixExt<Number>            &M)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  /**
   * Convert the matrix block \p M_b in full matrix format to rank-k matrix
   * format.
   */
  if (m == 0 || n == 0 || M.m() == 0 || M.n() == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero, do
       * not allocate memory for the component matrices of the rank-k matrix,
       * since there is no effective data.}
       */
    }
  else
    {
      /**
       * Extract the data for the submatrix defined on the block cluster
       * \f$\tau
       * \times \sigma\f$ from the full global matrix \p M.
       */
      LAPACKFullMatrixExt<Number> M_b(tau_index_range, sigma_index_range, M);

      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M_b(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M_b(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M_b(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M_b(0, 0)) ? -1.0 : 1.0);
                }
            }
        }
      else
        {
          rank        = M_b.rank_k_decompose(A, B);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const size_type                               fixed_rank_k,
  const LAPACKFullMatrixExt<Number>            &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  /**
   * Convert the matrix block \p M_b in full matrix format to rank-k
   * format.
   */
  if (m == 0 || n == 0 || fixed_rank_k == 0 || M.m() == 0 || M.n() == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero, do
       * not allocate memory for the component matrices of the rank-k matrix,
       * since there is no effective data.}
       */
    }
  else
    {
      /**
       * Extract the data for the submatrix block \f$b = \tau \times \sigma\f$
       * in the original matrix \p M.
       */
      LAPACKFullMatrixExt<Number> M_b(tau_index_range,
                                      sigma_index_range,
                                      M,
                                      M_tau_index_range,
                                      M_sigma_index_range);

      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M_b(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M_b(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M_b(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M_b(0, 0)) ? -1.0 : 1.0);
                }
            }
        }
      else
        {
          rank        = M_b.rank_k_decompose(fixed_rank_k, A, B);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const LAPACKFullMatrixExt<Number>            &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  Assert(is_subset(tau_index_range, M_tau_index_range), ExcInternalError());
  Assert(is_subset(sigma_index_range, M_sigma_index_range), ExcInternalError());

  /**
   * Convert the matrix block \p M_b in full matrix format to rank-k
   * format.
   */
  if (m == 0 || n == 0 || M.m() == 0 || M.n() == 0)
    {
      /**
       * \alert{If the size of the row or column dimension of \p M is zero, do
       * not allocate memory for the component matrices of the rank-k matrix,
       * since there is no effective data.}
       */
    }
  else
    {
      /**
       * Extract the data for the submatrix block \f$b = \tau \times \sigma\f$
       * in the original matrix \p M.
       */
      LAPACKFullMatrixExt<Number> M_b(tau_index_range,
                                      sigma_index_range,
                                      M,
                                      M_tau_index_range,
                                      M_sigma_index_range);

      if (m == 1 && n == 1)
        {
          /**
           * Handle the case when the source full matrix is a scalar.
           */
          if (std::abs(M_b(0, 0)) < std::numeric_limits<real_type>::epsilon())
            {
              /**
               * When the scalar value is zero. We do nothing here and the
               * rank-k matrix still has a dimension \f$1 \times 1\f$ with
               * zero rank and empty component matrices.
               */
            }
          else
            {
              /**
               * When the scalar value is non-zero, both component matrices
               * have the dimension \f$1 \times 1\f$ and the rank-k matrix has
               * rank 1.
               */
              A.reinit(1, 1);
              B.reinit(1, 1);
              formal_rank = 1;
              rank        = 1;

              if constexpr (numbers::NumberTraits<Number>::is_complex)
                {
                  A(0, 0) = std::sqrt(M_b(0, 0));
                  B(0, 0) = A(0, 0);
                }
              else
                {
                  A(0, 0) = std::sqrt(std::abs(M_b(0, 0)));
                  B(0, 0) = A(0, 0) * (std::signbit(M_b(0, 0)) ? -1.0 : 1.0);
                }
            }
        }
      else
        {
          rank        = M_b.rank_k_decompose(A, B);
          formal_rank = A.n();
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const size_type                               fixed_rank_k,
  const RkMatrix<Number>                       &M)
  : A()
  , B()
  // Before restriction, set the rank and formal rank of the current
  // rank-k matrix to be the same as \p M.
  , rank(M.rank)
  , formal_rank(M.formal_rank)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  if (formal_rank > 0 && m != 0 && n != 0 && M.m != 0 && M.n != 0)
    {
      /**
       * Only when the formal rank of the current rank-k matrix is larger than
       * zero, there will be actual initialization of component matrices and
       * further restriction operation
       */
      A.reinit(this->m, M.formal_rank);
      B.reinit(this->n, M.formal_rank);

      /**
       * Restrict the component matrix \p A of the original global rank-k matrix
       * \p M to the cluster \f$\tau\f$.
       */
      for (size_type i = 0; i < m; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              A(i, j) = M.A(tau_index_range[0] + i, j);
            }
        }

      /**
       * Restrict the component matrix \p B of the original global rank-k matrix
       * \p M to the cluster \f$\sigma\f$.
       */
      for (size_type i = 0; i < n; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              B(i, j) = M.B(sigma_index_range[0] + i, j);
            }
        }

      this->truncate_to_rank(fixed_rank_k);
    }
  else
    {
      rank        = 0;
      formal_rank = 0;
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const RkMatrix<Number>                       &M)
  : A()
  , B()
  // Before restriction, set the rank and formal rank of the current
  // rank-k matrix to be the same as \p M.
  , rank(M.rank)
  , formal_rank(M.formal_rank)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  if (formal_rank > 0 && m != 0 && n != 0 && M.m != 0 && M.n != 0)
    {
      /**
       * Only when the formal rank of the current rank-k matrix is larger than
       * zero, there will be actual initialization of component matrices and
       * further restriction operation
       */
      A.reinit(this->m, M.formal_rank);
      B.reinit(this->n, M.formal_rank);

      /**
       * Restrict the component matrix \p A of the original global rank-k matrix
       * \p M to the cluster \f$\tau\f$.
       */
      for (size_type i = 0; i < m; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              A(i, j) = M.A(tau_index_range[0] + i, j);
            }
        }

      /**
       * Restrict the component matrix \p B of the original global rank-k matrix
       * \p M to the cluster \f$\sigma\f$.
       */
      for (size_type i = 0; i < n; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              B(i, j) = M.B(sigma_index_range[0] + i, j);
            }
        }

      const size_type fixed_rank_k =
        std::min(static_cast<size_type>(std::min(this->m, this->n)), M.rank);
      this->truncate_to_rank(fixed_rank_k);
    }
  else
    {
      rank        = 0;
      formal_rank = 0;
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const size_type                               fixed_rank_k,
  const RkMatrix<Number>                       &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range)
  : A()
  , B()
  // Before restriction, set the rank and formal rank of the current
  // rank-k matrix to be the same as \p M.
  , rank(M.rank)
  , formal_rank(M.formal_rank)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  Assert(is_subset(tau_index_range, M_tau_index_range), ExcInternalError());
  Assert(is_subset(sigma_index_range, M_sigma_index_range), ExcInternalError());

  if (formal_rank > 0 && m != 0 && n != 0 && M.m != 0 && M.n != 0 &&
      fixed_rank_k > 0)
    {
      /**
       * Only when the formal rank of the current rank-k matrix is larger than
       * zero, there will be actual initialization of component matrices and
       * further restriction operation
       */
      A.reinit(this->m, M.formal_rank);
      B.reinit(this->n, M.formal_rank);

      /**
       * Restrict the component matrix \p A of the original local rank-k matrix
       * \p M to the cluster \f$\tau\f$.
       */
      for (size_type i = 0; i < m; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              A(i, j) = M.A(tau_index_range[0] - M_tau_index_range[0] + i, j);
            }
        }

      /**
       * Restrict the component matrix \p B of the original local rank-k matrix
       * \p M to the cluster \f$\sigma\f$.
       */
      for (size_type i = 0; i < n; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              B(i, j) =
                M.B(sigma_index_range[0] - M_sigma_index_range[0] + i, j);
            }
        }

      this->truncate_to_rank(fixed_rank_k);
    }
  else
    {
      rank        = 0;
      formal_rank = 0;
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const RkMatrix<Number>                       &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range)
  : A()
  , B()
  // Before restriction, set the rank and formal rank of the current
  // rank-k matrix to be the same as \p M.
  , rank(M.rank)
  , formal_rank(M.formal_rank)
  , m(tau_index_range[1] - tau_index_range[0])
  , n(sigma_index_range[1] - sigma_index_range[0])
{
  Assert(is_subset(tau_index_range, M_tau_index_range), ExcInternalError());
  Assert(is_subset(sigma_index_range, M_sigma_index_range), ExcInternalError());

  if (formal_rank > 0 && m != 0 && n != 0 && M.m != 0 && M.n != 0)
    {
      /**
       * Only when the formal rank of the current rank-k matrix is larger than
       * zero, there will be actual initialization of component matrices and
       * further restriction operation
       */
      A.reinit(this->m, M.formal_rank);
      B.reinit(this->n, M.formal_rank);

      /**
       * Restrict the component \p A of the original local rank-k matrix \p M to
       * the cluster \f$\tau\f$.
       */
      for (size_type i = 0; i < m; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              A(i, j) = M.A(tau_index_range[0] - M_tau_index_range[0] + i, j);
            }
        }

      /**
       * Restrict the component \p B of the original local rank-k matrix \p M to
       * the cluster \f$\sigma\f$.
       */
      for (size_type i = 0; i < n; i++)
        {
          for (size_type j = 0; j < M.formal_rank; j++)
            {
              B(i, j) =
                M.B(sigma_index_range[0] - M_sigma_index_range[0] + i, j);
            }
        }

      const size_type fixed_rank_k =
        std::min(static_cast<size_type>(std::min(this->m, this->n)), M.rank);
      this->truncate_to_rank(fixed_rank_k);
    }
  else
    {
      rank        = 0;
      formal_rank = 0;
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const LAPACKFullMatrixExt<Number> &A,
                           const LAPACKFullMatrixExt<Number> &B)
  : A(A)
  , B(B)
  , rank(std::min(std::min(A.m(), B.m()), A.n()))
  , formal_rank(A.n())
  , m(A.m())
  , n(B.m())
{
  /**
   * The formal rank of the rank-k matrix is equal to the number of columns of
   * \p A or \p B. Hence, we make an assertion about their equality.
   */
  AssertDimension(A.n(), B.n());
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const size_type         fixed_rank_k,
                           const RkMatrix<Number> &M1,
                           const RkMatrix<Number> &M2,
                           bool                    is_horizontal_split)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(0)
  , n(0)
{
  if (fixed_rank_k > 0)
    {
      if (is_horizontal_split)
        {
          // Vertical stacking the two submatrices.
          AssertDimension(M1.n, M2.n);

          m = M1.m + M2.m;
          n = M1.n;

          if (m > 0 && n > 0)
            {
              if (M1.rank == 0 && M2.rank == 0)
                {
                  /**
                   * When both submatrices have zero ranks, the agglomerated
                   * matrix also has a zero rank. And there is nothing to be
                   * done.
                   */
                }
              else if (M1.rank == 0)
                {
                  /**
                   * When \p M1 has zero rank, the agglomeration operation is
                   * only an embedding of \p M2.
                   */
                  A.reinit(m, M2.formal_rank);
                  B.reinit(n, M2.formal_rank);
                  A.fill(M2.A, m - M2.A.m(), 0);
                  B           = M2.B;
                  rank        = M2.rank;
                  formal_rank = M2.formal_rank;

                  truncate_to_rank(fixed_rank_k);
                }
              else if (M2.rank == 0)
                {
                  /**
                   * When \p M2 has zero rank, the agglomeration operation is
                   * only an embedding of \p M1.
                   */
                  A.reinit(m, M1.formal_rank);
                  B.reinit(n, M1.formal_rank);
                  A.fill(M1.A, 0, 0);
                  B           = M1.B;
                  rank        = M1.rank;
                  formal_rank = M1.formal_rank;

                  truncate_to_rank(fixed_rank_k);
                }
              else
                {
                  RkMatrix<Number> M1_embedded(m, n, M1.formal_rank);
                  M1_embedded.A.fill(M1.A, 0, 0);
                  M1_embedded.B = M1.B;

                  RkMatrix<Number> M2_embedded(m, n, M2.formal_rank);
                  M2_embedded.A.fill(M2.A, m - M2.A.m(), 0);
                  M2_embedded.B = M2.B;

                  /**
                   * Perform formatted addition of the two embedded matrices.
                   */
                  M1_embedded.add((*this), M2_embedded, fixed_rank_k);
                }
            }
        }
      else
        {
          // Horizontal stacking the two submatrices.
          AssertDimension(M1.m, M2.m);

          m = M1.m;
          n = M1.n + M2.n;

          if (m > 0 && n > 0)
            {
              if (M1.rank == 0 && M2.rank == 0)
                {
                  /**
                   * When both submatrices have zero ranks, the agglomerated
                   * matrix also has a zero rank. And there is nothing to be
                   * done.
                   */
                }
              else if (M1.rank == 0)
                {
                  /**
                   * When \p M1 has zero rank, the agglomeration operation is
                   * only an embedding of \p M2.
                   */
                  A.reinit(m, M2.formal_rank);
                  B.reinit(n, M2.formal_rank);
                  A = M2.A;
                  B.fill(M2.B, n - M2.B.m(), 0);
                  rank        = M2.rank;
                  formal_rank = M2.formal_rank;

                  truncate_to_rank(fixed_rank_k);
                }
              else if (M2.rank == 0)
                {
                  /**
                   * When \p M2 has zero rank, the agglomeration operation is
                   * only an embedding of \p M1.
                   */
                  A.reinit(m, M1.formal_rank);
                  B.reinit(n, M1.formal_rank);
                  A = M1.A;
                  B.fill(M1.B, 0, 0);
                  rank        = M1.rank;
                  formal_rank = M1.formal_rank;

                  truncate_to_rank(fixed_rank_k);
                }
              else
                {
                  RkMatrix<Number> M1_embedded(m, n, M1.formal_rank);
                  M1_embedded.A = M1.A;
                  M1_embedded.B.fill(M1.B, 0, 0);

                  RkMatrix<Number> M2_embedded(m, n, M2.formal_rank);
                  M2_embedded.A = M2.A;
                  M2_embedded.B.fill(M2.B, n - M2.B.m(), 0);

                  /**
                   * Perform formatted addition of the two embedded matrices.
                   */
                  M1_embedded.add((*this), M2_embedded, fixed_rank_k);
                }
            }
        }
    }
  else
    {
      /**
       * Here we only set the correct dimension of the agglomerated rank-k
       * matrix.
       */
      if (is_horizontal_split)
        {
          // Vertical stacking the two submatrices.
          AssertDimension(M1.n, M2.n);

          m = M1.m + M2.m;
          n = M1.n;
        }
      else
        {
          // Horizontal stacking the two submatrices.
          AssertDimension(M1.m, M2.m);

          m = M1.m;
          n = M1.n + M2.n;
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const size_type         fixed_rank_k,
                           const RkMatrix<Number> &M11,
                           const RkMatrix<Number> &M12,
                           const RkMatrix<Number> &M21,
                           const RkMatrix<Number> &M22,
                           const Number            rank_factor)
  : A()
  , B()
  , rank(0)
  , formal_rank(0)
  , m(M11.m + M21.m)
  , n(M11.n + M12.n)
{
  AssertDimension(M11.m, M12.m);
  AssertDimension(M21.m, M22.m);
  AssertDimension(M11.n, M21.n);
  AssertDimension(M12.n, M22.n);

  if (fixed_rank_k > 0 && m > 0 && n > 0)
    {
      if (M11.rank == 0 && M12.rank == 0 && M21.rank == 0 && M22.rank == 0)
        {
          /**
           * When all submatrices have zero ranks, the agglomerated matrix
           * also has a zero rank. And there is nothing to be done.
           */
        }
      else
        {
          /**
           * Create a rank-k matrix which is an embedding of the submatrix.
           * N.B. The embedding of a matrix does not change its rank.
           *
           * At first, each matrix block is embedded into the large matrix by
           * padding zeros to the component matrices \p A and \p B in a rank-k
           * matrix. \alert{N.B. When perform rank-k matrix embedding, the
           * formal rank instead of the real rank should be used since matrix
           * row copy operation will be performed in which the formal rank
           * will be involved as the number of columns.}
           *
           * Next, pairwise formatted addition will be applied successively to
           * the four embedded matrices to achieve agglomeration.
           */

          /**
           * Calculate the increased rank for pairwise formatted addition.
           */
          const size_type increased_rank =
            static_cast<size_type>(fixed_rank_k * rank_factor);

          /**
           * Count the total number of effective agglomeration operations to
           * be performed.
           */
          unsigned int total_number_of_agglomerations = 0;
          if (M11.rank > 0)
            {
              total_number_of_agglomerations++;
            }

          if (M12.rank > 0)
            {
              total_number_of_agglomerations++;
            }

          if (M21.rank > 0)
            {
              total_number_of_agglomerations++;
            }

          if (M22.rank > 0)
            {
              total_number_of_agglomerations++;
            }

          /**
           * Define a counter for the current agglomeration operation. When it
           * is 1, directly embed the rank-k submatrix into the current rank-k
           * matrix. When it is less than \p total_number_of_agglomerations and
           * larger than 1, perform a formatted addition with a higher rank
           * than the specified rank. When it is equal to \p
           * total_number_of_agglomerations, perform a formatted addition with
           * the specified rank.
           */
          unsigned int agglomeration_counter = 1;

          if (M11.rank > 0)
            {
              /**
               * Directly embed the matrix \p M11 into the current rank-k
               * matrix.
               */
              A.reinit(m, M11.formal_rank);
              B.reinit(n, M11.formal_rank);
              A.fill(M11.A, 0, 0);
              B.fill(M11.B, 0, 0);
              rank        = M11.rank;
              formal_rank = M11.formal_rank;

              agglomeration_counter++;
            }

          if (M12.rank > 0)
            {
              if (agglomeration_counter == 1)
                {
                  /**
                   * Directly embed the matrix \p M12 into the current rank-k
                   * matrix.
                   */
                  A.reinit(m, M12.formal_rank);
                  B.reinit(n, M12.formal_rank);
                  A.fill(M12.A, 0, 0);
                  B.fill(M12.B, n - M12.B.m(), 0);
                  rank        = M12.rank;
                  formal_rank = M12.formal_rank;
                }
              else
                {
                  /**
                   * Embed the matrix \p M12 then add it into the current rank-k
                   * matrix.
                   */
                  RkMatrix<Number> M12_embedded(m, n, M12.formal_rank);
                  M12_embedded.A.fill(M12.A, 0, 0);
                  M12_embedded.B.fill(M12.B, n - M12.B.m(), 0);

                  if (agglomeration_counter < total_number_of_agglomerations)
                    {
                      this->add(M12_embedded, increased_rank);
                    }
                  else
                    {
                      this->add(M12_embedded, fixed_rank_k);
                    }
                }

              agglomeration_counter++;
            }

          if (M21.rank > 0)
            {
              if (agglomeration_counter == 1)
                {
                  /**
                   * Directly embed the matrix \p M21 into the current rank-k
                   * matrix.
                   */
                  A.reinit(m, M21.formal_rank);
                  B.reinit(n, M21.formal_rank);
                  A.fill(M21.A, m - M21.A.m(), 0);
                  B.fill(M21.B, 0, 0);
                  rank        = M21.rank;
                  formal_rank = M21.formal_rank;
                }
              else
                {
                  /**
                   * Embed the matrix \p M21 then add it into the current rank-k
                   * matrix.
                   */
                  RkMatrix<Number> M21_embedded(m, n, M21.formal_rank);
                  M21_embedded.A.fill(M21.A, m - M21.A.m(), 0);
                  M21_embedded.B.fill(M21.B, 0, 0);

                  if (agglomeration_counter < total_number_of_agglomerations)
                    {
                      this->add(M21_embedded, increased_rank);
                    }
                  else
                    {
                      this->add(M21_embedded, fixed_rank_k);
                    }
                }

              agglomeration_counter++;
            }

          if (M22.rank > 0)
            {
              if (agglomeration_counter == 1)
                {
                  /**
                   * Directly embed the matrix \p M22 into the current rank-k
                   * matrix.
                   */
                  A.reinit(m, M22.formal_rank);
                  B.reinit(n, M22.formal_rank);
                  A.fill(M22.A, m - M22.A.m(), 0);
                  B.fill(M22.B, n - M22.B.m(), 0);
                  rank        = M22.rank;
                  formal_rank = M22.formal_rank;
                }
              else
                {
                  /**
                   * Embed the matrix \p M22 then add it into the current rank-k
                   * matrix.
                   */
                  RkMatrix<Number> M22_embedded(m, n, M22.formal_rank);
                  M22_embedded.A.fill(M22.A, m - M22.A.m(), 0);
                  M22_embedded.B.fill(M22.B, n - M22.B.m(), 0);

                  if (agglomeration_counter < total_number_of_agglomerations)
                    {
                      this->add(M22_embedded, increased_rank);
                    }
                  else
                    {
                      this->add(M22_embedded, fixed_rank_k);
                    }
                }
            }

          if (rank > fixed_rank_k)
            {
              truncate_to_rank(fixed_rank_k);
            }
        }
    }
}


template <typename Number>
RkMatrix<Number>::RkMatrix(const RkMatrix<Number> &matrix)
  : A(matrix.A)
  , B(matrix.B)
  , rank(matrix.rank)
  , formal_rank(matrix.formal_rank)
  , m(matrix.m)
  , n(matrix.n)
{}


template <typename Number>
RkMatrix<Number> &
RkMatrix<Number>::operator=(const RkMatrix<Number> &matrix)
{
  m           = matrix.m;
  n           = matrix.n;
  formal_rank = matrix.formal_rank;
  rank        = matrix.rank;
  A           = matrix.A;
  B           = matrix.B;

  return (*this);
}


template <typename Number>
void
RkMatrix<Number>::reinit(const size_type m,
                         const size_type n,
                         const size_type fixed_rank_k)
{
  if (m != 0 && n != 0 && fixed_rank_k != 0)
    {
      A.reinit(m, fixed_rank_k);
      B.reinit(n, fixed_rank_k);
      rank        = fixed_rank_k;
      formal_rank = fixed_rank_k;
    }
  else
    {
      A.reinit(0, 0);
      B.reinit(0, 0);
      rank        = 0;
      formal_rank = 0;
    }

  this->m = m;
  this->n = n;
}


template <typename Number>
typename RkMatrix<Number>::size_type
RkMatrix<Number>::get_m() const
{
  return m;
}


template <typename Number>
typename RkMatrix<Number>::size_type
RkMatrix<Number>::get_n() const
{
  return n;
}


template <typename Number>
typename RkMatrix<Number>::size_type
RkMatrix<Number>::get_rank() const
{
  return rank;
}


template <typename Number>
typename RkMatrix<Number>::size_type
RkMatrix<Number>::get_formal_rank() const
{
  return formal_rank;
}


template <typename Number>
LAPACKFullMatrixExt<Number> &
RkMatrix<Number>::get_A()
{
  return A;
}


template <typename Number>
const LAPACKFullMatrixExt<Number> &
RkMatrix<Number>::get_A() const
{
  return A;
}


template <typename Number>
LAPACKFullMatrixExt<Number> &
RkMatrix<Number>::get_B()
{
  return B;
}


template <typename Number>
const LAPACKFullMatrixExt<Number> &
RkMatrix<Number>::get_B() const
{
  return B;
}


template <typename Number>
bool
RkMatrix<Number>::convertToFullMatrix(LAPACKFullMatrixExt<Number> &matrix) const
{
  if (m != 0 && n != 0 && formal_rank > 0)
    {
      matrix.reinit(m, n);
      A.mHmult(matrix, B);

      return true;
    }
  else
    {
      matrix.reinit(m, n);

      return false;
    }
}


template <typename Number>
void
RkMatrix<Number>::restrictToFullMatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  LAPACKFullMatrixExt<Number>                  &matrix) const
{
  RkMatrix<Number> rkmatrix_restricted(tau_index_range,
                                       sigma_index_range,
                                       (*this));
  rkmatrix_restricted.convertToFullMatrix(matrix);
}


template <typename Number>
void
RkMatrix<Number>::restrictToFullMatrix(
  const std::array<types::global_dof_index, 2> &rk_tau_index_range,
  const std::array<types::global_dof_index, 2> &rk_sigma_index_range,
  LAPACKFullMatrixExt<Number>                  &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range) const
{
  Assert(is_subset(M_tau_index_range, rk_tau_index_range), ExcInternalError());
  Assert(is_subset(M_sigma_index_range, rk_sigma_index_range),
         ExcInternalError());

  RkMatrix<Number> rkmatrix_restricted(M_tau_index_range,
                                       M_sigma_index_range,
                                       (*this),
                                       rk_tau_index_range,
                                       rk_sigma_index_range);
  rkmatrix_restricted.convertToFullMatrix(M);
}


template <typename Number>
void
RkMatrix<Number>::print_formatted(std::ostream      &out,
                                  const unsigned int precision,
                                  const bool         scientific,
                                  const unsigned int width,
                                  const char        *zero_string,
                                  const double       denominator,
                                  const double       threshold) const
{
  out << "RkMatrix.dim=(" << m << "," << n << ")\n";
  out << "RkMatrix.rank=" << rank << "\n";
  out << "RkMatrix.formal_rank=" << formal_rank << "\n";
  out << "RkMatrix.A=\n";
  A.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);
  out << "RkMatrix.B=\n";
  B.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);
}


template <typename Number>
void
RkMatrix<Number>::print_formatted_to_mat(std::ostream      &out,
                                         const std::string &name,
                                         const unsigned int precision,
                                         const bool         scientific,
                                         const unsigned int width,
                                         const char        *zero_string,
                                         const double       denominator,
                                         const double       threshold) const
{
  out << "# name: " << name << "\n"
      << "# type: scalar struct\n"
      << "# ndims: 2\n"
      << "1 1\n"
      << "# length: 6\n";

  out << "# name: A\n"
      << "# type: matrix\n"
      << "# rows: " << A.m() << "\n"
      << "# columns: " << A.n() << "\n";

  A.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);

  out << "\n\n";

  out << "# name: B\n"
      << "# type: matrix\n"
      << "# rows: " << B.m() << "\n"
      << "# columns: " << B.n() << "\n";

  B.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);

  out << "\n\n";

  out << "# name: rank\n"
      << "# type: scalar\n"
      << rank << "\n";

  out << "\n\n";

  out << "# name: formal_rank\n"
      << "# type: scalar\n"
      << formal_rank << "\n";

  out << "\n\n";

  out << "# name: m\n"
      << "# type: scalar\n"
      << m << "\n";

  out << "\n\n";

  out << "# name: n\n"
      << "# type: scalar\n"
      << n << "\n";

  out << "\n\n";
}


template <typename Number>
void
RkMatrix<Number>::keep_first_n_columns(const size_type n)
{
  AssertIndexRange(n, formal_rank + 1);

  if (n < formal_rank)
    {
      A.keep_first_n_columns(n, true);
      B.keep_first_n_columns(n, true);

      formal_rank = n;
      rank        = n;
    }
  else
    {
      // When n == formal_rank, do nothing.
    }
}


template <typename Number>
void
RkMatrix<Number>::truncate_to_rank(size_type new_rank)
{
  if (rank > 0 && new_rank > 0)
    {
      /**
       * \alert{Only when the current rank-k matrix and the given new rank
       * have nonzero rank, it is meaningful to perform the rank truncation.}
       */

      /**
       * Work flow introduction: Use QR decomposition to perform the rank
       * truncation.
       */
      LAPACKFullMatrixExt<Number>                                    U, VT;
      std::vector<typename numbers::NumberTraits<Number>::real_type> Sigma_r;

      rank = LAPACKFullMatrixExt<Number>::reduced_svd_on_AxBH(
        A, B, U, Sigma_r, VT, new_rank);

      /**
       * \alert{Since the initial matrix may not have the correct rank, even
       * though its \p rank is larger than zero, its actual \p rank may be less
       * or even zero. Therefore, it is possible that the returned \p rank from
       * the above execution of \p reduced_svd_on_AxBH may be zero. Therefore,
       * it needs a special treatment here.}
       */
      if (rank > 0)
        {
          if (n < m)
            {
              /**
               * Adopt right associativity when the matrix is long.
               */
              A = U;
              VT.scale_rows(Sigma_r);
              VT.transpose(B);
            }
          else
            {
              /**
               * Adopt left associativity when the matrix is wide.
               */
              U.scale_columns(A, Sigma_r);
              VT.transpose(B);
            }

          /**
           * Update the formal rank.
           */
          formal_rank = A.n();
        }
      else
        {
          A.reinit(0, 0);
          B.reinit(0, 0);
          rank        = 0;
          formal_rank = 0;
        }
    }
  else
    {
      A.reinit(0, 0);
      B.reinit(0, 0);
      rank        = 0;
      formal_rank = 0;
    }
}


template <typename Number>
void
RkMatrix<Number>::truncate_to_rank(size_type                    new_rank,
                                   LAPACKFullMatrixExt<Number> &C,
                                   LAPACKFullMatrixExt<Number> &D)
{
  if (rank > 0 && new_rank > 0)
    {
      /**
       * \alert{Only when the current rank-k matrix and the given new rank
       * have nonzero rank, it is meaningful to perform the rank truncation.}
       */

      /**
       * Work flow introduction: Use QR decomposition to perform the rank
       * truncation.
       */
      LAPACKFullMatrixExt<Number>                                    U, VT;
      std::vector<typename numbers::NumberTraits<Number>::real_type> Sigma_r;

      rank = LAPACKFullMatrixExt<Number>::reduced_svd_on_AxBH(
        A, B, U, Sigma_r, VT, C, D, new_rank);

      /**
       * \alert{Since the initial matrix may not have the correct rank, even
       * though its \p rank is larger than zero, its actual \p rank may be less
       * or even zero. Therefore, it is possible that the returned \p rank from
       * the above execution of \p reduced_svd_on_AxBH may be zero. Therefore,
       * it needs a special treatment here.}
       */
      if (rank > 0)
        {
          if (n < m)
            {
              /**
               * Adopt right associativity when the matrix is long.
               */
              A = U;
              VT.scale_rows(Sigma_r);
              VT.transpose(B);
            }
          else
            {
              /**
               * Adopt left associativity when the matrix is wide.
               */
              U.scale_columns(A, Sigma_r);
              VT.transpose(B);
            }

          /**
           * Update the formal rank.
           */
          formal_rank = A.n();
        }
      else
        {
          A.reinit(0, 0);
          B.reinit(0, 0);
          rank        = 0;
          formal_rank = 0;
        }
    }
  else
    {
      if (rank == 0)
        {
          /**
           * If the original matrix's rank is zero, the error matrices should
           * be zero matrices.
           */
          C.reinit(0, 0);
          D.reinit(0, 0);
        }
      else
        {
          /**
           * If the original matrix's rank is nonzero but it is to be
           * truncated to zero rank, the error matrices should be directly
           * copied from \f$A\f$ and \f$B\f$.
           */
          C = A;
          D = B;

          /**
           * Balance the Frobenius norm of @p C and @p D.
           */
          balance_frobenius_norm(C, D);
        }

      A.reinit(0, 0);
      B.reinit(0, 0);
      rank        = 0;
      formal_rank = 0;
    }
}


template <typename Number>
void
RkMatrix<Number>::calc_rank_upper_bound()
{
  const size_type rank_A = A.rank();
  const size_type rank_B = B.rank();

  rank = std::min(std::min(rank_A, rank_B), rank);
}


template <typename Number>
void
RkMatrix<Number>::vmult(Vector<Number>       &y,
                        const Vector<Number> &x,
                        const bool            adding) const
{
  if (rank > 0)
    {
      /**
       * \alert{Only when the rank is larger than zero, the multiplication
       * will be performed.}
       */

      /**
       * The vector storing \f$B^T x\f$
       */
      Vector<Number> z(formal_rank);

      B.Hvmult(z, x);
      A.vmult(y, z, adding);
    }
  else
    {
      if (!adding)
        {
          /**
           * The result vector \f$y\f$ will be overwritten by zeros.
           */
          y.reinit(y.size());
        }
    }
}


template <typename Number>
void
RkMatrix<Number>::Cvmult(Vector<Number>       &y,
                         const Vector<Number> &x,
                         const bool            adding) const
{
  if (rank > 0)
    {
      if constexpr (numbers::NumberTraits<Number>::is_complex)
        {
          Vector<Number> z(formal_rank);
          B.Tvmult(z, x);
          A.Cvmult(y, z, adding);
        }
      else
        {
          this->vmult(y, x, adding);
        }
    }
  else
    {
      if (!adding)
        {
          /**
           * The result vector \f$y\f$ will be overwritten by zeros.
           */
          y.reinit(y.size());
        }
    }
}


template <typename Number>
void
RkMatrix<Number>::Tvmult(Vector<Number>       &y,
                         const Vector<Number> &x,
                         const bool            adding) const
{
  if (rank > 0)
    {
      /**
       * \alert{Only when the rank is larger than zero, the multiplication
       * will be performed.}
       */

      /**
       * The vector storing \f$A^T x\f$
       */
      Vector<Number> z(formal_rank);

      A.Tvmult(z, x);

      if constexpr (numbers::NumberTraits<Number>::is_complex)
        {
          B.Cvmult(y, z, adding);
        }
      else
        {
          B.vmult(y, z, adding);
        }
    }
  else
    {
      if (!adding)
        {
          /**
           * The result vector \f$y\f$ will be overwritten by zeros.
           */
          y.reinit(y.size());
        }
    }
}


template <typename Number>
void
RkMatrix<Number>::Hvmult(Vector<Number>       &y,
                         const Vector<Number> &x,
                         const bool            adding) const
{
  if constexpr (numbers::NumberTraits<Number>::is_complex)
    {
      if (rank > 0)
        {
          /**
           * \alert{Only when the rank is larger than zero, the multiplication
           * will be performed.}
           */

          /**
           * The vector storing \f$A^H x\f$
           */
          Vector<Number> z(formal_rank);

          A.Hvmult(z, x);
          B.vmult(y, z, adding);
        }
      else
        {
          if (!adding)
            {
              /**
               * The result vector \f$y\f$ will be overwritten by zeros.
               */
              y.reinit(y.size());
            }
        }
    }
  else
    {
      this->Tvmult(y, x, adding);
    }
}


template <typename Number>
void
RkMatrix<Number>::add(RkMatrix<Number> &M, const RkMatrix<Number> &M2) const
{
  if (rank == 0)
    {
      M = M2;
    }
  else if (M2.rank == 0)
    {
      M = (*this);
    }
  else
    {
      /**
       * Stack the components of \p A and \p B.
       */
      LAPACKFullMatrixExt<Number> A_new, B_new;
      A.hstack(A_new, M2.A);
      B.hstack(B_new, M2.B);

      M = RkMatrix<Number>(A_new, B_new);
    }
}


template <typename Number>
void
RkMatrix<Number>::add(RkMatrix<Number>       &M,
                      const Number            m2,
                      const RkMatrix<Number> &M2) const
{
  if (rank == 0)
    {
      M = M2;
      M.A *= m2;
    }
  else if (M2.rank == 0)
    {
      M = (*this);
    }
  else
    {
      /**
       * Stack the components of \p A and \p B.
       */
      LAPACKFullMatrixExt<Number> A_new, B_new;

      /**
       * Multiply the factor with \p M2.A.
       */
      LAPACKFullMatrixExt<Number> M2A_mul_m2(M2.A);
      M2A_mul_m2 *= m2;

      A.hstack(A_new, M2A_mul_m2);
      B.hstack(B_new, M2.B);

      M = RkMatrix<Number>(A_new, B_new);
    }
}


template <typename Number>
void
RkMatrix<Number>::add(const RkMatrix<Number> &M1)
{
  if (rank == 0)
    {
      (*this) = M1;
    }
  else if (M1.rank == 0)
    {
      // Do nothing.
    }
  else
    {
      /**
       * Stack the components of \p A and \p B.
       */
      LAPACKFullMatrixExt<Number> A_new, B_new;
      A.hstack(A_new, M1.A);
      B.hstack(B_new, M1.B);

      (*this) = RkMatrix<Number>(A_new, B_new);
    }
}


template <typename Number>
void
RkMatrix<Number>::add(const Number m1, const RkMatrix<Number> &M1)
{
  if (rank == 0)
    {
      (*this) = M1;
      A *= m1;
    }
  else if (M1.rank == 0)
    {
      // Do nothing.
    }
  else
    {
      /**
       * Stack the components of \p A and \p B.
       */
      LAPACKFullMatrixExt<Number> A_new, B_new;

      /**
       * Multiply the factor \p m1 with \p M1.A.
       */
      LAPACKFullMatrixExt<Number> M1A_mul_m1(M1.A);
      M1A_mul_m1 *= m1;

      A.hstack(A_new, M1A_mul_m1);
      B.hstack(B_new, M1.B);

      (*this) = RkMatrix<Number>(A_new, B_new);
    }
}


template <typename Number>
void
RkMatrix<Number>::add(RkMatrix<Number>       &M,
                      const RkMatrix<Number> &M2,
                      const size_type         fixed_rank_k) const
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(M, M2);
  M.truncate_to_rank(fixed_rank_k);
}


template <typename Number>
void
RkMatrix<Number>::add(RkMatrix<Number>            &M,
                      const RkMatrix<Number>      &M2,
                      const size_type              fixed_rank_k,
                      LAPACKFullMatrixExt<Number> &C,
                      LAPACKFullMatrixExt<Number> &D) const
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(M, M2);
  M.truncate_to_rank(fixed_rank_k, C, D);
}


template <typename Number>
void
RkMatrix<Number>::add(RkMatrix<Number>       &M,
                      const Number            m2,
                      const RkMatrix<Number> &M2,
                      const size_type         fixed_rank_k) const
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(M, m2, M2);
  M.truncate_to_rank(fixed_rank_k);
}


template <typename Number>
void
RkMatrix<Number>::add(RkMatrix<Number>            &M,
                      const Number                 m2,
                      const RkMatrix<Number>      &M2,
                      const size_type              fixed_rank_k,
                      LAPACKFullMatrixExt<Number> &C,
                      LAPACKFullMatrixExt<Number> &D) const
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(M, m2, M2);
  M.truncate_to_rank(fixed_rank_k, C, D);
}


template <typename Number>
void
RkMatrix<Number>::add(const RkMatrix<Number> &M1, const size_type fixed_rank_k)
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(M1);
  this->truncate_to_rank(fixed_rank_k);
}


template <typename Number>
void
RkMatrix<Number>::add(const RkMatrix<Number>      &M1,
                      const size_type              fixed_rank_k,
                      LAPACKFullMatrixExt<Number> &C,
                      LAPACKFullMatrixExt<Number> &D)
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(M1);
  this->truncate_to_rank(fixed_rank_k, C, D);
}


template <typename Number>
void
RkMatrix<Number>::add(const Number            m1,
                      const RkMatrix<Number> &M1,
                      const size_type         fixed_rank_k)
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(m1, M1);
  this->truncate_to_rank(fixed_rank_k);
}


template <typename Number>
void
RkMatrix<Number>::add(const Number                 m1,
                      const RkMatrix<Number>      &M1,
                      const size_type              fixed_rank_k,
                      LAPACKFullMatrixExt<Number> &C,
                      LAPACKFullMatrixExt<Number> &D)
{
  /**
   * Perform the addition via a simple juxtaposition of matrix components.
   * Then rank truncation is carried out.
   */
  this->add(m1, M1);
  this->truncate_to_rank(fixed_rank_k, C, D);
}


template <typename Number>
void
RkMatrix<Number>::assemble_from_rkmatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const RkMatrix<Number>                       &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range,
  const size_type                               fixed_rank_k)
{
  /**
   * \alert{When perform rank-k matrix embedding, the formal rank instead of
   * the real rank should be used since matrix row copy operation will be
   * performed in which the formal rank will be involved as the number of
   * columns.}
   */
  if (M.rank > 0 && M.m > 0 && M.n > 0)
    {
      /**
       * \alert{Only when the rank-k matrix \p M has a rank larger than zero,
       * the assembly operation will take place.}
       */
      RkMatrix<Number> M_embedded(m, n, M.formal_rank);
      M_embedded.A.fill_rows(tau_index_range, M.A, M_tau_index_range);
      M_embedded.B.fill_rows(sigma_index_range, M.B, M_sigma_index_range);

      this->add(M_embedded, fixed_rank_k);
    }
}


template <typename Number>
void
RkMatrix<Number>::assemble_from_rkmatrix(
  const std::array<types::global_dof_index, 2> &tau_index_range,
  const std::array<types::global_dof_index, 2> &sigma_index_range,
  const RkMatrix<Number>                       &M,
  const std::array<types::global_dof_index, 2> &M_tau_index_range,
  const std::array<types::global_dof_index, 2> &M_sigma_index_range,
  const size_type                               fixed_rank_k,
  LAPACKFullMatrixExt<Number>                  &C,
  LAPACKFullMatrixExt<Number>                  &D)
{
  /**
   * \alert{When perform rank-k matrix embedding, the formal rank instead of
   * the real rank should be used since matrix row copy operation will be
   * performed in which the formal rank will be involved as the number of
   * columns.}
   */
  if (M.rank > 0 && M.m > 0 && M.n > 0)
    {
      /**
       * \alert{Only when the rank-k matrix \p M has a rank larger than zero,
       * the assembly operation will take place.}
       */
      RkMatrix<Number> M_embedded(m, n, M.formal_rank);
      M_embedded.A.fill_rows(tau_index_range, M.A, M_tau_index_range);
      M_embedded.B.fill_rows(sigma_index_range, M.B, M_sigma_index_range);

      this->add(M_embedded, fixed_rank_k, C, D);
    }
  else
    {
      C.reinit(0, 0);
      D.reinit(0, 0);
    }
}


template <typename Number>
typename numbers::NumberTraits<Number>::real_type
RkMatrix<Number>::frobenius_norm(const size_type number_of_columns) const
{
  Assert(
    number_of_columns <= formal_rank,
    ExcMessage(
      "The effective number of columns required for calculating the Frobenius norm should be less than or equal to the formal rank!"));

  Number result = Number(0.);

  Vector<Number> ui(A.m());
  Vector<Number> uj(A.m());
  Vector<Number> vi(B.m());
  Vector<Number> vj(B.m());

  for (size_type i = 0; i < number_of_columns; i++)
    {
      A.get_column(i, ui);
      B.get_column(i, vi);

      for (size_type j = 0; j < number_of_columns; j++)
        {
          A.get_column(j, uj);
          B.get_column(j, vj);

          result += (ui * uj) * (vi * vj);
        }
    }

  if constexpr (numbers::NumberTraits<Number>::is_complex)
    return std::sqrt(std::real(result));
  else
    return std::sqrt(result);
}


template <typename Number>
typename numbers::NumberTraits<Number>::real_type
RkMatrix<Number>::frobenius_norm(void) const
{
  return frobenius_norm(formal_rank);
}


template <typename Number>
std::size_t
RkMatrix<Number>::memory_consumption() const
{
  return sizeof(*this) + (A.memory_consumption() - sizeof(A)) +
         (B.memory_consumption() - sizeof(B));
}


/**
 * Print an RkMatrix in Octave text data format.
 * @param out
 * @param name
 * @param values
 * @param precision
 * @param scientific
 * @param width
 * @param zero_string
 * @param denominator
 * @param threshold
 */
template <typename Number>
void
print_rkmatrix_to_mat(std::ostream           &out,
                      const std::string      &name,
                      const RkMatrix<Number> &values,
                      const unsigned int      precision   = 8,
                      const bool              scientific  = true,
                      const unsigned int      width       = 0,
                      const char             *zero_string = "0",
                      const double            denominator = 1.,
                      const double            threshold   = 0.)
{
  out << "# name: " << name << "\n"
      << "# type: scalar struct\n"
      << "# ndims: 2\n"
      << "1 1\n"
      << "# length: 6\n";

  out << "# name: A\n"
      << "# type: matrix\n"
      << "# rows: " << values.A.m() << "\n"
      << "# columns: " << values.A.n() << "\n";

  values.A.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);

  out << "\n\n";

  out << "# name: B\n"
      << "# type: matrix\n"
      << "# rows: " << values.B.m() << "\n"
      << "# columns: " << values.B.n() << "\n";

  values.B.print_formatted(
    out, precision, scientific, width, zero_string, denominator, threshold);

  out << "\n\n";

  out << "# name: rank\n"
      << "# type: scalar\n"
      << values.rank << "\n";

  out << "\n\n";

  out << "# name: formal_rank\n"
      << "# type: scalar\n"
      << values.formal_rank << "\n";

  out << "\n\n";

  out << "# name: m\n"
      << "# type: scalar\n"
      << values.m << "\n";

  out << "\n\n";

  out << "# name: n\n"
      << "# type: scalar\n"
      << values.n << "\n";

  out << "\n\n";
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_RKMATRIX_H_
