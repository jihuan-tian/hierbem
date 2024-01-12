/**
 * @file hblockmatrix_skew_symm_preconditioner.h
 * @brief Introduction of hblockmatrix_skew_symm_preconditioner.h
 *
 * @date 2022-11-25
 * @author Jihuan Tian
 */
#ifndef INCLUDE_HBLOCKMATRIX_SKEW_SYMM_PRECONDITIONER_H_
#define INCLUDE_HBLOCKMATRIX_SKEW_SYMM_PRECONDITIONER_H_

#include "hmatrix.h"
#include "linalg.h"

namespace HierBEM
{
  /**
   * Class for the preconditioner of the skew symmetric block \hmat. LU
   * factorization is applied to this matrix.
   *
   * The block \hmat has the structure
   * \f[
   * \begin{pmatrix}
   * M_{11}   & M_{12} \\
   * (M_{21}) & M_{22}
   * \end{pmatrix},
   * \f]
   *
   * where \f$M_{11}\f$ and \f$M_{22}\f$ are symmetric and positive definite,
   * the matrix block \f$M_{21}=-M_{12}^T\f$ is not stored.
   *
   * After the LU factorization, the matrix becomes
   *
   * \f[
   * \begin{pmatrix}
   * L_{11}   & U_{12} \\
   *          & L_{22}
   * \end{pmatrix}.
   * \f]
   */
  template <int spacedim, typename Number = double>
  class HBlockMatrixSkewSymmPreconditioner
  {
  public:
    using size_type = const typename Vector<Number>::size_type;

    /**
     * Default constructor
     */
    HBlockMatrixSkewSymmPreconditioner()
      : M11(nullptr)
      , M12(nullptr)
      , M22(nullptr)
    {}

    /**
     * Constructor for wrapping \hmat pointers.
     *
     * @param M11
     * @param M12
     * @param M22
     */
    HBlockMatrixSkewSymmPreconditioner(HMatrix<spacedim, Number> *M11,
                                       HMatrix<spacedim, Number> *M12,
                                       HMatrix<spacedim, Number> *M22)
      : M11(M11)
      , M12(M12)
      , M22(M22)
    {}

    /**
     * Copy constructor
     */
    HBlockMatrixSkewSymmPreconditioner(
      const HBlockMatrixSkewSymmPreconditioner<spacedim, Number> &block_mat);

    /**
     * Assignment operator
     */
    HBlockMatrixSkewSymmPreconditioner<spacedim, Number> &
    operator=(
      const HBlockMatrixSkewSymmPreconditioner<spacedim, Number> &block_mat);

    void
    compute_lu_factorization(const unsigned int fixed_rank);

    /**
     * Matrix-vector multiplication
     *
     * @param y
     * @param x
     */
    void
    vmult(Vector<Number> &y, const Vector<Number> &x) const;

  private:
    HMatrix<spacedim, Number> *M11;
    HMatrix<spacedim, Number> *M12;
    HMatrix<spacedim, Number> *M22;
  };


  template <int spacedim, typename Number>
  HBlockMatrixSkewSymmPreconditioner<spacedim, Number>::
    HBlockMatrixSkewSymmPreconditioner(
      const HBlockMatrixSkewSymmPreconditioner<spacedim, Number> &block_mat)
    : M11(block_mat.M11)
    , M12(block_mat.M12)
    , M22(block_mat.M22)
  {}


  template <int spacedim, typename Number>
  HBlockMatrixSkewSymmPreconditioner<spacedim, Number> &
  HBlockMatrixSkewSymmPreconditioner<spacedim, Number>::operator=(
    const HBlockMatrixSkewSymmPreconditioner<spacedim, Number> &block_mat)
  {
    M11 = block_mat.M11;
    M12 = block_mat.M12;
    M22 = block_mat.M22;

    return (*this);
  }

  template <int spacedim, typename Number>
  void
  HBlockMatrixSkewSymmPreconditioner<spacedim, Number>::
    compute_lu_factorization(const unsigned int fixed_rank)
  {
    // Compute Cholesky factorization for \f$M_{11}\f$ in situ, which produces
    // \f$L_{11}\f$.
    if (M11->get_type() == HMatrixType::HierarchicalMatrixType)
      {
        M11->compute_cholesky_factorization_task_parallel(fixed_rank);
      }
    else
      {
        M11->compute_cholesky_factorization(fixed_rank);
      }
    M11->solve_cholesky_by_forward_substitution_matrix_valued(*M12, fixed_rank);
    // Calculate \f$M_{22} = M_{22} + U_{12}^T U_{12}\f$
    M12->Tmmult_level_conserving(*M22, *M12, fixed_rank, true);
    // Apply Cholesky factorization to the new @p M22. After this operation,
    // @p M22 stores @p L22.
    if (M22->get_type() == HMatrixType::HierarchicalMatrixType)
      {
        M22->compute_cholesky_factorization_task_parallel(fixed_rank);
      }
    else
      {
        M22->compute_cholesky_factorization(fixed_rank);
      }
  }


  template <int spacedim, typename Number>
  void
  HBlockMatrixSkewSymmPreconditioner<spacedim, Number>::vmult(
    Vector<Number>       &x,
    const Vector<Number> &b) const
  {
    // Verify the dimensions of matrices and vectors should match.
    AssertDimension(x.size(), M11->get_n() + M12->get_n());
    AssertDimension(b.size(), M11->get_m() + M22->get_m());

    // Verify the dimensions of matrix blocks should match.
    AssertDimension(M11->get_m(), M12->get_m());
    AssertDimension(M12->get_n(), M22->get_n());

    // Verify the whole block matrix should be square.
    AssertDimension(x.size(), b.size());

    // Split the vectors.
    const size_type n1 = M11->get_n();
    const size_type n2 = M12->get_n();
    Vector<Number>  b1(n1);
    Vector<Number>  b2(n2);

    copy_vector(b1, 0, b, 0, n1);
    copy_vector(b2, 0, b, n1, n2);

    // Solve \f$Ly=b\f$.
    Vector<Number> y1(n1);
    Vector<Number> y2(n2);

    M11->solve_cholesky_by_forward_substitution(y1, b1);
    M12->Tvmult(b2, y1);
    M22->solve_cholesky_by_forward_substitution(y2, b2);

    // Solve \f$Ux=y\f$.
    Vector<Number> x1(n1);
    Vector<Number> x2(n2);
    M22->solve_cholesky_by_backward_substitution(x2, y2);
    M12->vmult(y1, -1.0, x2);
    M11->solve_cholesky_by_backward_substitution(x1, y1);

    // Merge the result vector.
    copy_vector(x, 0, x1, 0, n1);
    copy_vector(x, n1, x2, 0, n2);
  }
} // namespace HierBEM

#endif /* INCLUDE_HBLOCKMATRIX_SKEW_SYMM_PRECONDITIONER_H_ */
