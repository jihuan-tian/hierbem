/**
 * \file lapack_linalg.h
 * \brief Linear algebra computation using LAPACK functions.
 * \date 2021-06-10
 * \author Jihuan Tian
 */

#ifndef INCLUDE_LAPACK_FULL_MATRIX_EXT_H_
#define INCLUDE_LAPACK_FULL_MATRIX_EXT_H_

#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "general_exceptions.h"
#include "generic_functors.h"
#include "lapack_helpers.h"

#define MAYBE_UNUSED(x) [&x] {}()

namespace HierBEM
{
  using namespace dealii;

  /**
   * Exception for invalid LAPACKFullMatrix state (@p LAPACKSupport::State).
   */
  DeclException1(ExcInvalidLAPACKFullMatrixState,
                 LAPACKSupport::State,
                 << "Invalid LAPACKFullMatrix state: " << arg1);

  /**
   * Exception for invalid LAPACKFullMatrix property (@p LAPACKSupport::Property).
   */
  DeclException1(ExcInvalidLAPACKFullMatrixProperty,
                 LAPACKSupport::Property,
                 << "Invalid LAPACKFullMatrix property: " << arg1);

  /**
   * Calculate singular value threshold value which can be used for estimating
   * the matrix's rank and perform rank truncation.
   *
   * The calculation is according to <code>tol = max (size (A)) * Sigma_r(1) *
   * epsilon;</code>, where \p epsilon is the machine's precision.
   * @param Sigma_r list of singular values organized in decreasing order.
   * @return
   */
  template <typename Number = double>
  Number
  calc_singular_value_threshold(
    const size_t m,
    const size_t n,
    const std::vector<typename numbers::NumberTraits<Number>::real_type>
      &Sigma_r)
  {
    return std::max(m, n) * Sigma_r[0] * std::numeric_limits<double>::epsilon();
  }


  /**
   * Extend and expose more of the the functionality of LAPACKFullMatrix.
   */
  template <typename Number>
  class LAPACKFullMatrixExt : public LAPACKFullMatrix<Number>
  {
  public:
    /**
     * Declare the type for container size.
     */
    using size_type = std::make_unsigned<types::blas_int>::type;

    // Friend function declaration.
    /**
     * Balance the Frobenius norm of the two matrices.
     *
     * @param A
     * @param B
     */
    template <typename Number1>
    friend void
    balance_frobenius_norm(LAPACKFullMatrixExt<Number1> &A,
                           LAPACKFullMatrixExt<Number1> &B);

    /**
     * Create a zero valued matrix.
     *
     * @param rows
     * @param cols
     * @param matrix
     */
    static void
    ZeroMatrix(const size_type              rows,
               const size_type              cols,
               LAPACKFullMatrixExt<Number> &matrix);

    /**
     * Create constant valued matrix.
     *
     * @param rows
     * @param cols
     * @param value
     * @param matrix
     */
    static void
    ConstantMatrix(const size_type              rows,
                   const size_type              cols,
                   Number                       value,
                   LAPACKFullMatrixExt<Number> &matrix);

    /**
     * Create a constant valued diagonal matrix.
     *
     * @param dim
     * @param value
     * @param matrix
     */
    static void
    DiagMatrix(const size_type dim, Number value, LAPACKFullMatrixExt &matrix);

    /**
     * Create an identity matrix of dimension \p dim.
     * @param dim
     * @param matrix
     */
    static void
    IdentityMatrix(const size_type dim, LAPACKFullMatrixExt &matrix);

    /**
     * Reshape a vector of values into a LAPACKFullMatrixExt in column major.
     */
    static void
    Reshape(const size_type              rows,
            const size_type              cols,
            const std::vector<Number>   &values,
            LAPACKFullMatrixExt<Number> &matrix);

    /**
     * Perform SVD on the product of two component matrices \f$A\f$ and
     * \f$B^T\f$ without rank truncation. If the matrix is not of full rank,
     * truncate it to the effective rank. It returns the effective rank.
     *
     * \alert{The operation of this function has no accuracy loss.}
     * @param A
     * @param B
     * @param U
     * @param Sigma_r
     * @param VT
     */
    static size_type
    reduced_svd_on_AxBT(
      LAPACKFullMatrixExt<Number>                                    &A,
      LAPACKFullMatrixExt<Number>                                    &B,
      LAPACKFullMatrixExt<Number>                                    &U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number>                                    &VT,
      Number singular_value_threshold = 0.);

    /**
     * Perform SVD on the product of two component matrices \f$A\f$ and
     * \f$B^T\f$ with truncation by a specified rank.
     *
     * \alert{The operation of this function may have accuracy loss when the
     * specified rank is less than the actual rank of the matrix.}
     *
     * N.B. If the actual rank of the matrix is less than the specified \p
     * truncation_rank, i.e. rank deficient, the matrix will be truncated to its
     * actual rank. Then, this function has no accuracy loss.
     * @param A
     * @param B
     * @param U
     * @param Sigma_r
     * @param VT
     * @param truncation_rank
     */
    static size_type
    reduced_svd_on_AxBT(
      LAPACKFullMatrixExt<Number>                                    &A,
      LAPACKFullMatrixExt<Number>                                    &B,
      LAPACKFullMatrixExt<Number>                                    &U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number>                                    &VT,
      size_type truncation_rank,
      Number    singular_value_threshold = 0.);

    /**
     * Perform SVD on the product of two component matrices \f$A\f$ and
     * \f$B^T\f$ with truncation by a specified rank. In addition, the
     * components \f$C\f$ and \f$D\f$ of the error matrix due to rank truncation
     * are returned. The error matrix \f$E = CD^T\f$.
     *
     * \alert{The operation of this function may have accuracy loss when the
     * specified rank is less than the actual rank of the matrix.}
     *
     * N.B. If the actual rank of the matrix is less than the specified \p
     * truncation_rank, i.e. rank deficient, the matrix will be truncated to its
     * actual rank. Then, this function has no accuracy loss.
     * @param A
     * @param B
     * @param U
     * @param Sigma_r
     * @param VT
     * @param C
     * @param D
     * @param truncation_rank
     */
    static size_type
    reduced_svd_on_AxBT(
      LAPACKFullMatrixExt<Number>                                    &A,
      LAPACKFullMatrixExt<Number>                                    &B,
      LAPACKFullMatrixExt<Number>                                    &U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number>                                    &VT,
      LAPACKFullMatrixExt<Number>                                    &C,
      LAPACKFullMatrixExt<Number>                                    &D,
      size_type truncation_rank,
      Number    singular_value_threshold = 0.);

    /**
     * Construct a square matrix by specifying the dimension.
     */
    LAPACKFullMatrixExt(const size_type size = 0);

    /**
     * Construct a matrix by specifying the number of rows and columns.
     * @param rows
     * @param cols
     */
    LAPACKFullMatrixExt(const size_type rows, const size_type cols);

    /**
     * Copy constructor from an LAPACKFullMatrixExt object.
     * @param mat
     */
    LAPACKFullMatrixExt(const LAPACKFullMatrixExt &mat);

    /**
     * Copy constructor from an LAPACKFullMatrix object.
     *
     * \mynote{N.B. The @p state and @p property fields of @p mat is not
     * accessible, because they are private members in the class
     * @p LAPACKFullMatrix<Number>. Hence the @p state of the current matrix is
     * set to the default @p matrix and its @p property is set to the default
     * @p general.}
     *
     * @param mat
     */
    LAPACKFullMatrixExt(const LAPACKFullMatrix<Number> &mat);

    /**
     * Construct a full matrix by restriction to the block cluster \f$\tau
     * \times \sigma\f$ defined by their index ranges from the global full matrix \p M.
     *
     * @param row_index_range
     * @param column_index_range
     * @param M
     */
    LAPACKFullMatrixExt(
      const std::array<types::global_dof_index, 2> &row_index_range,
      const std::array<types::global_dof_index, 2> &column_index_range,
      const LAPACKFullMatrixExt<Number>            &M);

    /**
     * Construct a full matrix by restriction to the block cluster \f$\tau
     * \times \sigma\f$ defined by their index ranges from the local full matrix \p M.
     *
     * @param row_index_range
     * @param column_index_range
     * @param M The larger full matrix to be restricted.
     * @param M_row_index_range
     * @param M_column_index_range
     */
    LAPACKFullMatrixExt(
      const std::array<types::global_dof_index, 2> &row_index_range,
      const std::array<types::global_dof_index, 2> &column_index_range,
      const LAPACKFullMatrixExt<Number>            &M,
      const std::array<types::global_dof_index, 2> &M_row_index_range,
      const std::array<types::global_dof_index, 2> &M_column_index_range);

    /**
     * Construct a full matrix \f$M\f$ from an agglomeration of two full
     * submatrices \f$M_1\f$ and \f$M_2\f$, which have been obtained from either
     * horizontal splitting or vertical splitting.
     *
     * When the two submatrices have been obtained from horizontal splitting,
     * @p vstack will be used for the agglomeration. When the two submatrices have
     * been obtained from vertical splitting, \p hstack will be used for the
     * agglomeration.
     *
     * @param M1
     * @param M2
     * @param is_horizontal_split
     */
    LAPACKFullMatrixExt(const LAPACKFullMatrixExt &M1,
                        const LAPACKFullMatrixExt &M2,
                        bool                       is_horizontal_split);

    /**
     * Construct a full matrix \f$M\f$ from an agglomeration of four
     * full submatrices, \f$M_{11}, M_{12}, M_{21}, M_{22}\f$.
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
     *   <dd>
     * 1. This method implements (7.7) for full matrices in Hackbusch's
     * \f$\mathcal{H}\f$-matrix book.
     * 2. This method is only applicable in the case when the cardinality based
     * cluster tree partition is used.
     *   </dd>
     * </dl>
     */
    LAPACKFullMatrixExt(const LAPACKFullMatrixExt &M11,
                        const LAPACKFullMatrixExt &M12,
                        const LAPACKFullMatrixExt &M21,
                        const LAPACKFullMatrixExt &M22);

    /**
     * Overloaded assignment operator.
     * @param matrix
     * @return
     */
    LAPACKFullMatrixExt<Number> &
    operator=(const LAPACKFullMatrixExt<Number> &matrix);

    /**
     * Overloaded assignment operator.
     * @param matrix
     * @return
     */
    LAPACKFullMatrixExt<Number> &
    operator=(const LAPACKFullMatrix<Number> &matrix);

    /**
     * Assign a scalar value to all elements of the matrix.
     *
     * @param a
     * @return
     */
    LAPACKFullMatrixExt<Number> &
    operator=(const Number d);

    void
    reinit(const size_type nrows, const size_type ncols);

    /**
     * Set a matrix column as zeros.
     */
    void
    set_column_zeros(const size_type col_index);

    /**
     * Set a matrix row as zeros.
     */
    void
    set_row_zeros(const size_type row_index);

    /**
     * Get the values of the row \p row_index into a \p Vector.
     * @param row_index
     * @param row_values
     */
    void
    get_row(const size_type row_index, Vector<Number> &row_values) const;

    /**
     * Get the values of the column \p col_index into a \p Vector.
     * @param col_index
     * @param col_values
     */
    void
    get_column(const size_type col_index, Vector<Number> &col_values) const;

    /**
     * Remove row \p row_index from the matrix.
     * @param row_index
     */
    void
    remove_row(const size_type row_index);

    /**
     * Remove rows \p row_indices from the matrix.
     * @param row_indices
     */
    void
    remove_rows(const std::vector<size_type> &row_indices);

    /**
     * Keep only the first n rows of the matrix.
     *
     * When \p other_rows_removed is true, the other rows are deleted from the
     * matrix, otherwise they are set to zero.
     *
     * @param n
     * @param other_rows_removed
     */
    void
    keep_first_n_rows(const size_type n, bool other_rows_removed = true);

    /**
     * Remove column \p column_index from the matrix.
     * @param column_index
     */
    void
    remove_column(const size_type column_index);

    /**
     * Remove columns \p column_indices from the matrix.
     * @param column_indices
     */
    void
    remove_columns(const std::vector<size_type> &column_indices);

    /**
     * Keep only the first n columns of the matrix.
     *
     * When \p other_columns_removed is true, the other columns are deleted from
     * the matrix, otherwise they are set to zero.
     *
     * @param n
     * @param other_rows_removed
     */
    void
    keep_first_n_columns(const size_type n, bool other_columns_removed = true);

    /**
     * Perform the standard singular value decomposition (SVD).
     *
     * @param U with a dimension \f$m \times m\f$.
     * @param Sigma_r the list of singular values, with a dimension \f$\min(m,n)\f$.
     * @param VT with a dimension \f$n \times n\f$
     */
    void
    svd(LAPACKFullMatrixExt<Number>                                    &U,
        std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
        LAPACKFullMatrixExt<Number>                                    &VT);

    /**
     * Perform the standard singular value decomposition (SVD) with rank
     * truncation.
     *
     * When the given \p truncation_rank is less than the minimum dimension
     * \f$\min{m, n}\f$, \p U's \p (truncation_rank+1)'th to \p m'th columns are
     * set to zeros; \p VT's \p (truncation_rank+1)'th to \p n'th rows are set to
     * zeros; \p Sigma_r's \p (truncation_rank+1)'th to \p n'th values are set to
     * zeros.
     * @param U with a dimension \f$m \times m\f$.
     * @param Sigma_r the list of singular values, which has a dimension \f$\min(m,n)\f$.
     * @param VT with a dimension \f$n \times n\f$.
     */
    void
    svd(LAPACKFullMatrixExt<Number>                                    &U,
        std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
        LAPACKFullMatrixExt<Number>                                    &VT,
        const size_type truncation_rank);

    /**
     * Perform the reduced singular value decomposition (SVD) with rank
     * truncation.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>Even though an explicit truncation rank is specified, inside this
     * function, after SVD, the effective rank of the matrix is obtained.
     * Because any given truncation rank value larger than the effective matrix
     * rank is meaningless, it will be limited to be the effective rank.</dd>
     * </dl>
     *
     * @param U
     * @param Sigma_r
     * @param VT
     * @param truncation_rank Truncation rank specified by the user.
     * @param singular_value_threshold
     * @return Effective rank.
     */
    size_type
    reduced_svd(
      LAPACKFullMatrixExt<Number>                                    &U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number>                                    &VT,
      size_type truncation_rank,
      Number    singular_value_threshold = 0.);

    /**
     * Perform the reduced singular value decomposition (SVD) with rank
     * truncation. In addition, the components \f$C\f$ and \f$D\f$ of the error
     * matrix due to rank truncation are returned. The error matrix \f$E =
     * CD^T\f$.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>Even though an explicit truncation rank is specified, inside this
     * function, after SVD, the effective rank of the matrix is obtained.
     * Because any given truncation rank value larger than the effective matrix
     * rank is meaningless, it will be limited to be the effective rank.</dd>
     * </dl>
     *
     * @param U
     * @param Sigma_r
     * @param VT
     * @param C
     * @param D
     * @param truncation_rank Truncation rank specified by the user.
     * @param singular_value_threshold
     * @return Effective rank.
     */
    size_type
    reduced_svd(
      LAPACKFullMatrixExt<Number>                                    &U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number>                                    &VT,
      LAPACKFullMatrixExt<Number>                                    &C,
      LAPACKFullMatrixExt<Number>                                    &D,
      size_type truncation_rank,
      Number    singular_value_threshold = 0.);

    /**
     * Compute the LU factorization of the full matrix.
     *
     * \mynote{This is a copy of the same function in \p LAPACKFullMatrix<Number>
     * in order that the permutation vector \p ipiv can be accessed, since \p
     * ipiv is private in \p LAPACKFullMatrix<Number>.}
     */
    void
    compute_lu_factorization();

    /**
     * Compute the Cholesky factorization of the full matrix.
     *
     * \mynote{This is a wrapper of the same function in \p LAPACKFullMatrix<Number>
     * in order that the \p state of the current \p LAPACKFullMatrixExt<Number>
     * object can be updated. Before the factorization, the full matrix has
     * @p matrix state and @p symmetric property. After the factorization, it has
     * @p cholesky state and @p lower_triangular property.}
     */
    void
    compute_cholesky_factorization();

    /**
     * Get the reference to the vector which stores the row permutation relation
     * obtained from an LU factorization.
     *
     * @return The vector storing the row permutation relation. Its i'th element
     * stores the row index of the matrix to which the i'th row migrates.
     */
    std::vector<types::blas_int> &
    get_lu_permutation();

    /**
     * Get the const reference to the vector which stores the row permutation
     * relation obtained from an LU factorization.
     *
     * @return The vector storing the row permutation relation. Its i'th element
     * stores the row index of the matrix to which the i'th row migrates.
     */
    const std::vector<types::blas_int> &
    get_lu_permutation() const;

    /**
     * Get the rank of the matrix using SVD.
     *
     * A copy will be made at first for SVD to operate on.
     * @param threshold threshold for singular values. The number of singular
     * values larger than this threshold is the matrix rank.
     * @return
     */
    size_type
    rank(Number threshold = 0.) const;

    /**
     * Perform QR decomposition of the matrix.
     * @param Q
     * @param R
     */
    void
    qr(LAPACKFullMatrixExt<Number> &Q, LAPACKFullMatrixExt<Number> &R);

    /**
     * Perform reduced QR decomposition of the matrix.
     * @param Q
     * @param R
     */
    void
    reduced_qr(LAPACKFullMatrixExt<Number> &Q, LAPACKFullMatrixExt<Number> &R);

    /**
     * Left multiply the current matrix with a diagonal matrix \p V, which is
     * equivalent to scale the rows of the current matrix with the corresponding
     * scalar elements in @p V. The result is stored in a @p std::vector.
     *
     * @param V
     */
    void
    scale_rows(
      const std::vector<typename numbers::NumberTraits<Number>::real_type> &V);

    /**
     * Left multiply the current matrix with a diagonal matrix \p V, which is
     * equivalent to scale the rows of the current matrix with the corresponding
     * scalar elements in @p V. The result is stored in a new matrix @p A.
     *
     * @param A
     * @param V
     */
    void
    scale_rows(
      LAPACKFullMatrixExt<Number>                                          &A,
      const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
      const;

    /**
     * Right multiply the current matrix with a diagonal matrix \p V, which is
     * equivalent to scale the columns of the current matrix with the
     * corresponding scalar elements in @p V. The result is stored in a
     * @p std::vector.
     *
     * @param V
     */
    void
    scale_columns(
      const std::vector<typename numbers::NumberTraits<Number>::real_type> &V);

    /**
     * Right multiply the current matrix with a diagonal matrix \p V, which is
     * equivalent to scale the columns of the current matrix with the
     * corresponding scalar elements in @p V. The result is stored in a new
     * matrix @p A.
     *
     * @param A
     * @param V
     */
    void
    scale_columns(
      LAPACKFullMatrixExt<Number>                                          &A,
      const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
      const;

    /**
     * Right multiply the current matrix with a diagonal matrix \p V, which is
     * equivalent to scale the columns of the current matrix with the
     * corresponding scalar elements in @p V. The result is stored in a
     * @p Vector.
     *
     * @param V
     */
    void
    scale_columns(
      const Vector<typename numbers::NumberTraits<Number>::real_type> &V);

    /**
     * Perform in-place transpose of the matrix.
     */
    void
    transpose();

    /**
     * Get the transpose of the current matrix into a new matrix \p AT.
     * @param AT
     */
    void
    transpose(LAPACKFullMatrixExt<Number> &AT) const;

    /**
     * Fill a rectangular block into the current matrix. This function has the
     * similar behavior as FullMatrix::fill.
     *
     * @param src
     * @param dst_offset_i
     * @param dst_offset_j
     * @param src_offset_i
     * @param src_offset_j
     * @param factor
     * @param transpose
     */
    template <typename MatrixType>
    void
    fill(const MatrixType &src,
         const size_type   dst_offset_i = 0,
         const size_type   dst_offset_j = 0,
         const size_type   src_offset_i = 0,
         const size_type   src_offset_j = 0,
         const Number      factor       = 1.,
         const bool        transpose    = false,
         const bool        is_adding    = false);

    /**
     * Fill the \p values to the \p row_index'th row of the current matrix.
     * @param row_index
     * @param values
     */
    void
    fill_row(const size_type       row_index,
             const Vector<Number> &values,
             const bool            is_adding = false);

    /**
     * Fill a specified row in the destination matrix with a row extracted from
     * the source matrix.
     *
     * @param dst_row_index
     * @param src_row_index
     * @param M
     * @param factor
     * @param is_adding
     */
    void
    fill_row(const size_type                    dst_row_index,
             const size_type                    src_row_index,
             const LAPACKFullMatrixExt<Number> &M,
             const Number                       factor    = 1.,
             const bool                         is_adding = false);

    /**
     * Fill the data in \p M by rows into the current matrix based on the their
     * index ranges.
     */
    void
    fill_rows(const std::array<types::global_dof_index, 2> &row_index_range,
              const LAPACKFullMatrixExt<Number>            &M,
              const std::array<types::global_dof_index, 2> &M_row_index_range,
              const Number                                  factor    = 1.,
              const bool                                    is_adding = false);

    /**
     * Fill the \p values to the \p col_index'th column of the current matrix.
     * @param col_index
     * @param values
     */
    void
    fill_col(const size_type       col_index,
             const Vector<Number> &values,
             const bool            is_adding = false);

    /**
     * Horizontally stack two matrices, \f$C = [A, B]\f$.
     * @param C
     * @param B
     */
    void
    hstack(LAPACKFullMatrixExt<Number>       &C,
           const LAPACKFullMatrixExt<Number> &B) const;

    /**
     * Vertically stack two matrices, \f$C = [A; B]\f$.
     * @param C
     * @param B
     */
    void
    vstack(LAPACKFullMatrixExt<Number>       &C,
           const LAPACKFullMatrixExt<Number> &B) const;

    /**
     * Decompose the full matrix into the two components of its rank-k
     * representation, the associativity of triple-matrix multiplication is
     * automatically detected.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>This method will be used for converting a full matrix to a rank-k
     * matrix, which underlies the operator \f$\mathcal{T}_{r}^{\mathcal{R}
     * \leftarrow \mathcal{F}}\f$ in (7.2) in Hackbusch's
     * \f$\mathcal{H}\f$-matrix book.</dd>
     * </dl>
     * @param k
     * @param A
     * @param B
     * @return
     */
    size_type
    rank_k_decompose(const unsigned int           k,
                     LAPACKFullMatrixExt<Number> &A,
                     LAPACKFullMatrixExt<Number> &B);

    /**
     * Decompose the full matrix into the two components of its rank-k
     * representation, the associativity of triple-matrix multiplication is
     * automatically detected. The error matrices are returned.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>This method will be used for converting a full matrix to a rank-k
     * matrix, which underlies the operator \f$\mathcal{T}_{r}^{\mathcal{R}
     * \leftarrow \mathcal{F}}\f$ in (7.2) in Hackbusch's
     * \f$\mathcal{H}\f$-matrix book.</dd>
     * </dl>
     * @param k
     * @param A
     * @param B
     * @param C
     * @param D
     * @return
     */
    size_type
    rank_k_decompose(const unsigned int           k,
                     LAPACKFullMatrixExt<Number> &A,
                     LAPACKFullMatrixExt<Number> &B,
                     LAPACKFullMatrixExt<Number> &C,
                     LAPACKFullMatrixExt<Number> &D);

    /**
     * Decompose the full matrix into the two components of its rank-k
     * representation, the associativity of triple-matrix multiplication is
     * automatically detected. This version does not have an actual rank
     * truncation but sets the truncation rank to be the minimum matrix
     * dimension \f$\min\{m, n\}\f$.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>This method will be used for converting a full matrix to a rank-k
     * matrix, which implements the operator \f$\mathcal{T}_{r}^{\mathcal{R}
     * \leftarrow \mathcal{F}}\f$ in (7.2) in Hackbusch's
     * \f$\mathcal{H}\f$-matrix book.</dd>
     * </dl>
     * @param A
     * @param B
     * @return Effective rank of the matrix.
     */
    size_type
    rank_k_decompose(LAPACKFullMatrixExt<Number> &A,
                     LAPACKFullMatrixExt<Number> &B);

    /**
     * Decompose the full matrix into the two components of its rank-k
     * representation.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>Because the \p reduced_svd member function is called by \p
     * rank_k_decompose, the effective rank of the matrix will be returned,
     * which
     * can be smaller than the specified fixed rank \p k.</dd>
     * </dl>
     *
     * @param k User specified truncation rank.
     * @param is_left_associative
     * @param A
     * @param B
     * @return Effective rank.
     */
    size_type
    rank_k_decompose(const unsigned int           k,
                     LAPACKFullMatrixExt<Number> &A,
                     LAPACKFullMatrixExt<Number> &B,
                     bool                         is_left_associative);

    /**
     * Decompose the full matrix into the two components of its rank-k
     * representation, while the error matrices are returned.
     *
     * <dl class="section note">
     *   <dt>Note</dt>
     *   <dd>Because the \p reduced_svd member function is called by \p
     * rank_k_decompose, the effective rank of the matrix will be returned,
     * which
     * can be smaller than the specified fixed rank \p k.</dd>
     * </dl>
     *
     * @param k User specified truncation rank.
     * @param is_left_associative
     * @param A
     * @param B
     * @param C
     * @param D
     * @return Effective rank.
     */
    size_type
    rank_k_decompose(const unsigned int           k,
                     LAPACKFullMatrixExt<Number> &A,
                     LAPACKFullMatrixExt<Number> &B,
                     LAPACKFullMatrixExt<Number> &C,
                     LAPACKFullMatrixExt<Number> &D,
                     bool                         is_left_associative);

    /**
     * Add two matrix into a new matrix \f$C = A + B\f$, where \f$A\f$ is the
     * current matrix.
     */
    void
    add(LAPACKFullMatrixExt<Number>       &C,
        const LAPACKFullMatrixExt<Number> &B,
        const bool is_result_matrix_symm_apriori = false) const;

    /**
     * Add two matrix into a new matrix \f$C = A + b*B\f$ with a factor \f$b\f$
     * multiplied with \f$B\f$, where \f$A\f$ is the current matrix.
     */
    void
    add(LAPACKFullMatrixExt<Number>       &C,
        const Number                       b,
        const LAPACKFullMatrixExt<Number> &B,
        const bool is_result_matrix_symm_apriori = false) const;

    /**
     * Add the matrix \p B into the current matrix.
     * @param B
     */
    void
    add(const LAPACKFullMatrixExt<Number> &B,
        const bool is_result_matrix_store_tril_only = false);

    /**
     * Add the matrix \p B into the current matrix with a factor \f$b\f$
     * multiplied with \f$B\f$.
     * @param B
     */
    void
    add(const Number                       b,
        const LAPACKFullMatrixExt<Number> &B,
        const bool is_result_matrix_store_tril_only = false);

    /**
     * Matrix-vector multiplication which also handles the case when the matrix
     * is symmetric and lower triangular.
     *
     * \mynote{When the matrix is symmetric, the LAPACK function @p symv is
     * adopted. In my implementation, only those lower triangular entries in a
     * symmetric full matrix are used by @p symv.
     *
     * At the moment, there is no counterpart @p Tvmult implemented which handles
     * the case of symmetric matrix, because there is no difference between
     * \f$Av\f$ and \f$A^T v\f$.}
     *
     * @param w
     * @param v
     * @param adding
     */
    void
    vmult(Vector<Number>       &w,
          const Vector<Number> &v,
          const bool            adding = false) const;

    /**
     * Transposed matrix-vector multiplication which also handles the case when
     * the matrix is symmetric.
     *
     * \mynote{1. When the matrix is symmetric, this function simply calls
     * @p LAPACKFullMatrixExt::vmult, because there is no difference between
     * \f$Av\f$ and \f$A^T v\f$.
     * 2. When the matrix is general, it calls the member function in the member
     * class @p LAPACKFullMatrix::Tvmult.}
     *
     * @param w
     * @param v
     * @param adding
     */
    void
    Tvmult(Vector<Number>       &w,
           const Vector<Number> &v,
           const bool            adding = false) const;

    /**
     * Multiply two matrices, i.e. \f$C = AB\f$ or \f$C = C + AB\f$.
     *
     * @param C
     * @param B
     * @param adding
     */
    void
    mmult(LAPACKFullMatrixExt<Number>       &C,
          const LAPACKFullMatrixExt<Number> &B,
          const bool                         adding = false) const;

    /**
     * Multiply two matrices with the product scaled by a factor, i.e. \f$C =
     * \alpha \cdot AB\f$ or \f$C = C + \alpha \cdot AB\f$.
     *
     * @param C
     * @param alpha
     * @param B
     * @param adding
     */
    void
    mmult(LAPACKFullMatrixExt<Number>       &C,
          const Number                       alpha,
          const LAPACKFullMatrixExt<Number> &B,
          const bool                         adding = false) const;

    /**
     * Multiply two matrices with the second operand transposed, i.e. \f$C =
     * AB^T\f$ or \f$C = C + AB^T\f$.
     *
     * @param C
     * @param B
     * @param adding
     */
    void
    mTmult(LAPACKFullMatrixExt<Number>       &C,
           const LAPACKFullMatrixExt<Number> &B,
           const bool                         adding = false) const;

    /**
     * Multiply two matrices with the second operand transposed and with the
     * product scaled by a factor, i.e. \f$C = \alpha \cdot AB^T\f$ or \f$C = C
     * + \alpha \cdot AB^T\f$.
     *
     * @param C
     * @param alpha
     * @param B
     * @param adding
     */
    void
    mTmult(LAPACKFullMatrixExt<Number>       &C,
           const Number                       alpha,
           const LAPACKFullMatrixExt<Number> &B,
           const bool                         adding = false) const;

    void
    Tmmult(LAPACKFullMatrixExt<Number>       &C,
           const LAPACKFullMatrixExt<Number> &B,
           const bool                         adding = false) const;

    void
    Tmmult(LAPACKFullMatrixExt<Number>       &C,
           const Number                       alpha,
           const LAPACKFullMatrixExt<Number> &B,
           const bool                         adding = false) const;

    /**
     * Calculate the determinant of a \f$2\times 2\f$ matrix.
     *
     * @return
     */
    Number
    determinant2x2() const;

    /**
     * Calculate the determinant of a \f$3\times 3\f$ matrix.
     *
     * @return
     */
    Number
    determinant3x3() const;

    /**
     * Assign the inverse of the given matrix to @p *this.
     *
     * @param M
     */
    void
    invert(const LAPACKFullMatrixExt<Number> &M);

    /**
     * Calculate the inverse of the matrix using Gauss elimination.
     *
     * @param M_inv
     */
    void
    invert_by_gauss_elim(LAPACKFullMatrixExt<Number> &M_inv);

    /**
     * Set the matrix property of type \p LAPACKSupport::Property for the matrix.
     *
     * \comment{The property should be explicitly set by the caller and the
     * algorithm will not check its validity, since in most cases, this is
     * difficult or expensive to achieve.}
     *
     * @param property
     */
    void
    set_property(const LAPACKSupport::Property property);

    /**
     * Set the matrix state.
     * @param state
     */
    void
    set_state(const LAPACKSupport::State state);

    /**
     * Get the matrix property.
     *
     * @return
     */
    LAPACKSupport::Property
    get_property() const;

    /**
     * Get the matrix state.
     * @return
     */
    LAPACKSupport::State
    get_state() const;

    /**
     * Solve the linear system with right hand side @p v and put the solution
     * back to @p v. The matrix should be either triangular or LU/Cholesky
     * factorization should be previously computed.
     *
     * The flag transposed indicates whether the solution of the transposed
     * system is to be performed.
     *
     * \mynote{This function is copied from @p lapack_full_matrix.cc, so that the
     * matrix state held by @p LAPACKFullMatrixExt can be accessed.}
     *
     * @param v
     * @param transposed
     */
    void
    solve(Vector<Number> &v, const bool transposed = false) const;

    /**
     * Solve the unit lower triangular matrix \f$Lx=b\f$ by forward
     * substitution.
     *
     * The right hand side vector \f$b\f$ will be overwritten by the solution
     * vector \f$x\f$.
     *
     * \comment{When the forward substitution procedure is applied to the matrix
     * factorized by LU, the right hand side vector \p b should be permuted
     * according to the \p ipiv vector, which is obtained from the LAPACK LU
     * factorization with row pivoting. Remember that we have this relation
     * \f$PA=LU\f$.
     *
     * However, when the lower triangular matrix to be solved is the transpose
     * of an upper triangular matrix, there is no need to permute the right hand
     * side
     * vector \p b anymore, because permutation has been or will be performed
     * during the solution of matrix blocks related to the lower triangulation
     * matrix. For short, the permutation of the right hand side vector should
     * only be performed once.
     *
     * By the way, no permutation is needed when calling backward substitution
     * or solving Cholesky related matrices.}
     *
     * @param b Right hand side vector and after execution, it stores the result vector.
     * @param transposed
     * @param permute_rhs_vector
     */
    void
    solve_by_forward_substitution(Vector<Number> &b,
                                  const bool      transposed       = false,
                                  const bool      is_unit_diagonal = true,
                                  const bool permute_rhs_vector = false) const;

    /**
     * Solve the unit lower triangular matrix \f$Lx=b\f$ by forward
     * substitution.
     *
     * @param x Result vector.
     * @param b Right hand side vector.
     * @param transposed
     */
    void
    solve_by_forward_substitution(Vector<Number>       &x,
                                  const Vector<Number> &b,
                                  const bool            transposed = false,
                                  const bool is_unit_diagonal = true) const;

    /**
     * Solve the upper triangular matrix \f$Ux=b\f$ by backward substitution.
     *
     * The right hand side vector \f$b\f$ will be overwritten by the solution
     * vector \f$x\f$.
     *
     * @param b Right hand side vector and after execution, it stores the result vector.
     * @param transposed
     */
    void
    solve_by_backward_substitution(Vector<Number> &b,
                                   const bool      transposed  = false,
                                   const bool is_unit_diagonal = false) const;

    /**
     * Solve the upper triangular matrix \f$Ux=b\f$ by backward substitution.
     *
     * @param x Result vector.
     * @param b Right hand side vector.
     * @param transposed
     */
    void
    solve_by_backward_substitution(Vector<Number>       &x,
                                   const Vector<Number> &b,
                                   const bool            transposed = false,
                                   const bool is_unit_diagonal = false) const;

    /**
     * Print a LAPACKFullMatrixExt to Octave mat format.
     *
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
     * Read the data of a full matrix with the specified variable name from a
     * file
     * saved by Octave using the option \p -text.
     *
     * @param in Input file stream
     * @param name Variable name
     */
    void
    read_from_mat(std::ifstream &in, const std::string &name);

    /**
     * Determine an estimate for the memory consumption (in bytes) of this
     * object.
     *
     * @return
     */
    std::size_t
    memory_consumption() const;

    /**
     * Determine the memory consumption for the core data stored in this object,
     * which is calculated by counting the number of data values then multiplied
     * by the size of the value type.
     *
     * @return
     */
    std::size_t
    memory_consumption_for_core_data() const;

  private:
    LAPACKSupport::State    state;
    LAPACKSupport::Property property;

    /**
     * The scalar factors of the elementary reflectors.
     */
    std::vector<typename numbers::NumberTraits<Number>::real_type> tau;

    /**
     * Work space used by LAPACK routines.
     */
    std::vector<Number> work;

    /**
     * Integer work array used by LAPACK routines.
     */
    std::vector<types::blas_int> iwork;

    /**
     * The vector storing the permutations applied for pivoting in the
     * LU-factorization.
     */
    std::vector<types::blas_int> ipiv;
  };


  template <typename Number>
  void
  balance_frobenius_norm(LAPACKFullMatrixExt<Number> &A,
                         LAPACKFullMatrixExt<Number> &B)
  {
    Number norm_A = A.frobenius_norm();
    Number norm_B = B.frobenius_norm();
    Number factor = std::sqrt(norm_A / norm_B);

    if (factor < std::numeric_limits<double>::epsilon())
      {
        /**
         * The factor is almost zero, do nothing to the matrix @p A.
         */
      }
    else
      {
        A /= factor;
      }

    B *= factor;
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::ZeroMatrix(const size_type              rows,
                                          const size_type              cols,
                                          LAPACKFullMatrixExt<Number> &matrix)
  {
    matrix.reinit(rows, cols);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::ConstantMatrix(
    const size_type              rows,
    const size_type              cols,
    Number                       value,
    LAPACKFullMatrixExt<Number> &matrix)
  {
    matrix.reinit(rows, cols);

    for (size_type j = 0; j < cols; j++)
      {
        for (size_type i = 0; i < rows; i++)
          {
            matrix(i, j) = value;
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::DiagMatrix(const size_type      dim,
                                          Number               value,
                                          LAPACKFullMatrixExt &matrix)
  {
    matrix.reinit(dim, dim);

    for (size_type i = 0; i < dim; i++)
      {
        matrix(i, i) = value;
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::IdentityMatrix(const size_type      dim,
                                              LAPACKFullMatrixExt &matrix)
  {
    matrix.reinit(dim, dim);

    for (size_type i = 0; i < dim; i++)
      {
        matrix(i, i) = 1.;
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::Reshape(const size_type              rows,
                                       const size_type              cols,
                                       const std::vector<Number>   &values,
                                       LAPACKFullMatrixExt<Number> &matrix)
  {
    AssertDimension(rows * cols, values.size());

    matrix.reinit(rows, cols);

    typename std::vector<Number>::const_iterator it_values = values.begin();
    for (typename LAPACKFullMatrixExt<Number>::iterator it = matrix.begin();
         it != matrix.end();
         it++, it_values++)
      {
        (*it) = (*it_values);
      }
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::reduced_svd_on_AxBT(
    LAPACKFullMatrixExt<Number>                                    &A,
    LAPACKFullMatrixExt<Number>                                    &B,
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT,
    Number singular_value_threshold)
  {
    /**
     * <dl class="section">
     *   <dt>Work flow</dt>
     *   <dd>
     */

    /**
     * In a rank-k matrix, the number of columns in the component matrix \p A,
     * which is the formal rank, should match that of \p B. So we make an
     * assertion here.
     */
    AssertDimension(A.n(), B.n());

    const size_type mm      = A.m();
    const size_type nn      = B.m();
    const size_type min_dim = std::min(mm, nn);

    bool is_qr_used;

    /**
     * N.B. The number of columns in the component matrix \p A or \p B is the
     * formal rank of the rank-k matrix, which means the actual rank of the
     * rank-k matrix is less than this formal rank.
     */
    const size_type formal_rank = A.n();

    if (A.m() > formal_rank && B.m() > formal_rank)
      {
        is_qr_used = true;

        /**
         * When both \p A and \p B are long matrices, i.e. they have more rows
         * than columns, perform the reduced QR decomposition to component
         * matrix
         * \p A, which has a dimension of \f$m \times r\f$.
         */
        LAPACKFullMatrixExt<Number> QA, RA;
        A.reduced_qr(QA, RA);

        /**
         * Perform the reduced QR decomposition to component matrix \p B, which
         * has a dimension of \f$n \times r\f$.
         */
        LAPACKFullMatrixExt<Number> QB, RB;
        B.reduced_qr(QB, RB);

        /**
         * Perform SVD to the product \f$R\f$ of the two upper triangular
         * matrices, i.e. \f$R = R_A R_B^T\f$.
         */
        LAPACKFullMatrixExt<Number> R, U_hat, VT_hat;
        /**
         * N.B. Before LAPACK matrix multiplication, the memory of the result
         * matrix should be reinitialized.
         */
        R.reinit(RA.m(), RB.m());
        RA.mTmult(R, RB);
        R.svd(U_hat, Sigma_r, VT_hat);

        U.reinit(QA.m(), formal_rank);
        QA.mmult(U, U_hat);
        VT.reinit(formal_rank, QB.m());
        VT_hat.mTmult(VT, QB);
      }
    else
      {
        is_qr_used = false;

        /**
         * When \p A or \p B is not a long matrix, firstly convert the rank-k
         * matrix to a full matrix, then perform SVD on this full matrix.
         */
        LAPACKFullMatrixExt<Number> fullmatrix(A.m(), B.m());
        A.mTmult(fullmatrix, B);
        fullmatrix.svd(U, Sigma_r, VT);
      }

    if (singular_value_threshold == 0.)
      {
        /**
         * If the singular value threshold is perfect zero, calculate a
         * threshold value instead.
         */
        singular_value_threshold =
          calc_singular_value_threshold(mm, nn, Sigma_r);
      }

    /**
     * Get the actual rank of the matrix by counting the total number of
     * singular values which are larger than the given threshold \p
     * singular_value_threshold. The actual rank should always be less than or
     * equal to \p min_dim. The matrix will be truncated to this effective rank.
     */
    size_type rank = 0;
    for (size_t i = 0; i < Sigma_r.size(); i++)
      {
        if (Sigma_r[i] > singular_value_threshold)
          {
            rank++;
          }
      }
    AssertIndexRange(rank, min_dim + 1);

    if (rank < min_dim)
      {
        /**
         * When the matrix is not of full rank, keep the first \p rank singular
         * values, while discarding the others.
         */
        std::vector<typename numbers::NumberTraits<Number>::real_type> copy(
          std::move(Sigma_r));
        Sigma_r.resize(rank);
        for (size_type i = 0; i < rank; i++)
          {
            Sigma_r.at(i) = copy.at(i);
          }

        /**
         * Keep the first \p rank columns of \p U, while deleting the others.
         */
        U.keep_first_n_columns(rank, true);

        /**
         * Keep the first \p rank rows of \p VT, while deleting the others.
         */
        VT.keep_first_n_rows(rank, true);
      }
    else
      {
        /**
         * When the matrix is of full rank, i.e. \p rank == \p min_dim.
         */
        AssertDimension(rank, min_dim);

        if (is_qr_used)
          {
            /**
             * When QR decomposition has been used, if \p M has a dimension \f$m
             * \times n\f$, the dimensions of all matrices are:
             * * \f$U \in \mathbb{R}^{m \times {\rm formal rank}}\f$
             * * \f$\Sigma_r \in \mathbb{R}^{{\rm formal rank} \times {\rm
             * formal rank}}\f$
             * * \f$V \in \mathbb{R}^{n \times {\rm formal rank}}\f$
             */
            if (rank < formal_rank)
              {
                /**
                 * When the matrix rank is less than the formal rank, keep the
                 * first \p rank singular values, while discarding the others.
                 */
                std::vector<typename numbers::NumberTraits<Number>::real_type>
                  copy(std::move(Sigma_r));
                Sigma_r.resize(rank);
                for (size_type i = 0; i < rank; i++)
                  {
                    Sigma_r.at(i) = copy.at(i);
                  }

                /**
                 * Keep the first \p rank columns of \p U, while deleting the
                 * others.
                 */
                U.keep_first_n_columns(rank, true);

                /**
                 * Keep the first \p rank rows of \p VT, while deleting the
                 * others.
                 */
                VT.keep_first_n_rows(rank, true);
              }
          }
        else
          {
            /**
             * When QR decomposition has not been used, the results \p U, \p
             * Sigma_r and \p VT are obtained from full matrix SVD. And if \p M
             * has a dimension \f$m \times n\f$, the dimensions of all matrices
             * obtained from SVD are:
             * * \f$U \in \mathbb{R}^{m \times m}\f$
             * * \f$\Sigma_r \in \mathbb{R}^{m \times n}\f$
             * * \f$V \in \mathbb{R}^{n \times n}\f$
             */
            if (mm > nn)
              {
                /**
                 * When \p M is a long matrix, \f$\Sigma_r =
                 * \begin{pmatrix}\Sigma_r' \\ 0 \end{pmatrix}\f$. Therefore, we
                 * keep the first \p min_dim columns of \p U, while deleting the
                 * others. \p VT is kept intact.
                 */
                U.keep_first_n_columns(min_dim, true);
              }
            else if (mm < nn)
              {
                /**
                 * When \p M is a wide matrix, \f$\Sigma_r = \begin{pmatrix}
                 * \Sigma_r' & 0 \end{pmatrix}\f$. Therefore, we keep the first
                 * \p
                 * min_dim rows of \p VT, while deleting the others. \p U is kept
                 * intact.
                 */
                VT.keep_first_n_rows(min_dim, true);
              }
            else
              {
                /**
                 * When \p M is square, do nothing.
                 */
              }
          }
      }

    AssertDimension(U.n(), Sigma_r.size());
    AssertDimension(VT.m(), Sigma_r.size());

    /**
     * Return the actual rank of the matrix.
     */
    return rank;

    /**
     *   </dd>
     * </dl>
     */
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::reduced_svd_on_AxBT(
    LAPACKFullMatrixExt<Number>                                    &A,
    LAPACKFullMatrixExt<Number>                                    &B,
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT,
    size_type truncation_rank,
    Number    singular_value_threshold)
  {
    /**
     * In a rank-k matrix, the number of columns in the component matrix \p A
     * should match that of \p B.
     */
    AssertDimension(A.n(), B.n());

    const size_type mm      = A.m();
    const size_type nn      = B.m();
    const size_type min_dim = std::min(mm, nn);

    bool is_qr_used;

    /**
     * N.B. The number of columns in the component matrix \p A or \p B is the
     * formal rank of the rank-k matrix, which means the rank-k matrix may
     * not be of full rank and the actual rank is less than this formal
     * rank.
     */
    const size_type formal_rank = A.n();

    if (A.m() > formal_rank && B.m() > formal_rank)
      {
        is_qr_used = true;

        /**
         * When both \p A and \p B are long matrices, i.e. they have more rows
         * than columns, perform reduced QR decomposition to component matrix \p
         * A, which has a dimension of \f$m \times r\f$.
         */
        LAPACKFullMatrixExt<Number> QA, RA;
        A.reduced_qr(QA, RA);

        /**
         * Perform reduced QR decomposition to component matrix \p B, which
         * has a dimension of \f$n \times r\f$.
         */
        LAPACKFullMatrixExt<Number> QB, RB;
        B.reduced_qr(QB, RB);

        /**
         * Perform SVD to the product \f$R\f$ of the two upper triangular
         * matrices, i.e. \f$R = R_A R_B^T\f$.
         */
        LAPACKFullMatrixExt<Number> R, U_hat, VT_hat;
        /**
         * N.B. Before LAPACK matrix multiplication, the memory of the result
         * matrix should be reinitialized.
         */
        R.reinit(RA.m(), RB.m());
        RA.mTmult(R, RB);
        R.svd(U_hat, Sigma_r, VT_hat);

        U.reinit(QA.m(), formal_rank);
        QA.mmult(U, U_hat);
        VT.reinit(formal_rank, QB.m());
        VT_hat.mTmult(VT, QB);
      }
    else
      {
        is_qr_used = false;

        /**
         * Firstly convert the rank-k matrix to a full matrix, then perform
         * SVD on this full matrix.
         */
        LAPACKFullMatrixExt<Number> fullmatrix(A.m(), B.m());
        A.mTmult(fullmatrix, B);
        fullmatrix.svd(U, Sigma_r, VT);
      }

    if (singular_value_threshold == 0.)
      {
        /**
         * If the singular value threshold is perfect zero, calculate a
         * threshold value instead.
         */
        singular_value_threshold =
          calc_singular_value_threshold(mm, nn, Sigma_r);
      }

    /**
     * Get the actual rank of the matrix and it should always be less than or
     * equal to \p min_dim.
     */
    size_type rank = 0;
    for (size_t i = 0; i < Sigma_r.size(); i++)
      {
        if (Sigma_r[i] > singular_value_threshold)
          {
            rank++;
          }
      }
    AssertIndexRange(rank, min_dim + 1);

    /**
     * Limit the truncation rank wrt. the actual rank.
     */
    if (truncation_rank > rank)
      {
        truncation_rank = rank;
      }

    if (truncation_rank < min_dim)
      {
        if (truncation_rank > 0)
          {
            /**
             * Keep the first \p truncation_rank singular values, while discarding
             * others.
             */
            std::vector<typename numbers::NumberTraits<Number>::real_type> copy(
              std::move(Sigma_r));
            Sigma_r.resize(truncation_rank);
            for (size_type i = 0; i < truncation_rank; i++)
              {
                Sigma_r.at(i) = copy.at(i);
              }

            /**
             * Keep the first \p truncation_rank columns of \p U, while deleting
             * others.
             */
            U.keep_first_n_columns(truncation_rank, true);

            /**
             * Keep the first \p truncation_rank rows of \p VT, while deleting
             * others.
             */
            VT.keep_first_n_rows(truncation_rank, true);
          }
        else
          {
            /**
             * When the truncation rank is zero, clear the SVD result matrix and
             * singular value vector.
             */
            U.reinit(0, 0);
            VT.reinit(0, 0);
            Sigma_r.clear();
          }
      }
    else
      {
        /**
         * In this case, it could only be \p truncation_rank == \p min_dim.
         */
        AssertDimension(truncation_rank, min_dim);

        if (is_qr_used)
          {
            /**
             * In this case, the results \p U, \p Sigma_r and \p VT are obtained
             * via QR decomposition. And if \p M has a dimension \f$m \times n\f$,
             * the dimensions of all matrices are:
             * * \f$U \in \mathbb{R}^{m \times {\rm formal rank}}\f$
             * * \f$\Sigma_r \in \mathbb{R}^{{\rm formal rank} \times {\rm
             * formal rank}}\f$
             * * \f$V \in \mathbb{R}^{n \times {\rm formal rank}}\f$
             */
            if (truncation_rank < formal_rank)
              {
                /**
                 * Keep the first \p truncation_rank singular values, while
                 * discarding others.
                 */
                std::vector<typename numbers::NumberTraits<Number>::real_type>
                  copy(std::move(Sigma_r));
                Sigma_r.resize(truncation_rank);
                for (size_type i = 0; i < truncation_rank; i++)
                  {
                    Sigma_r.at(i) = copy.at(i);
                  }

                /**
                 * Keep the first \p truncation_rank columns of \p U, while
                 * deleting others.
                 */
                U.keep_first_n_columns(truncation_rank, true);

                /**
                 * Keep the first \p truncation_rank rows of \p VT, while deleting
                 * others.
                 */
                VT.keep_first_n_rows(truncation_rank, true);
              }
          }
        else
          {
            /**
             * In this case, the results \p U, \p Sigma_r and \p VT are obtained
             * from full matrix SVD. And if \p M has a dimension \f$m \times
             * n\f$, the dimensions of all matrices obtained from SVD are:
             * * \f$U \in \mathbb{R}^{m \times m}\f$
             * * \f$\Sigma_r \in \mathbb{R}^{m \times n}\f$
             * * \f$V \in \mathbb{R}^{n \times n}\f$
             */
            if (mm > nn)
              {
                /**
                 * When the original matrix is long, \f$\Sigma_r =
                 * \begin{pmatrix}\Sigma_r' \\ 0 \end{pmatrix}\f$. Therefore, we
                 * keep the first \p min_dim columns of \p U, while deleting
                 * others. \p VT is kept intact.
                 */
                U.keep_first_n_columns(min_dim, true);
              }
            else if (mm < nn)
              {
                /**
                 * When the original matrix is wide, \f$\Sigma_r =
                 * \begin{pmatrix} \Sigma_r' & 0 \end{pmatrix}\f$. Therefore, we
                 * keep the first \p
                 * min_dim rows of \p VT, while deleting others. \p U is kept
                 * intact.
                 */
                VT.keep_first_n_rows(min_dim, true);
              }
            else
              {
                /**
                 * When the original matrix is square, do nothing.
                 */
              }
          }
      }

    AssertDimension(U.n(), Sigma_r.size());
    AssertDimension(VT.m(), Sigma_r.size());

    return truncation_rank;
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::reduced_svd_on_AxBT(
    LAPACKFullMatrixExt<Number>                                    &A,
    LAPACKFullMatrixExt<Number>                                    &B,
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT,
    LAPACKFullMatrixExt<Number>                                    &C,
    LAPACKFullMatrixExt<Number>                                    &D,
    size_type truncation_rank,
    Number    singular_value_threshold)
  {
    /**
     * In a rank-k matrix, the number of columns in the component matrix \p A
     * should match that of \p B.
     */
    AssertDimension(A.n(), B.n());

    const size_type mm      = A.m();
    const size_type nn      = B.m();
    const size_type min_dim = std::min(mm, nn);

    bool is_qr_used;

    /**
     * N.B. The number of columns in the component matrix \p A or \p B is the
     * formal rank of the rank-k matrix, which means the rank-k matrix may
     * not be of full rank and the actual rank is less than this formal
     * rank.
     */
    const size_type formal_rank = A.n();

    if (A.m() > formal_rank && B.m() > formal_rank)
      {
        is_qr_used = true;

        /**
         * When both \p A and \p B are long matrices, i.e. they have more rows
         * than columns, i.e. the formal rank, perform reduced QR decomposition
         * to
         * component matrix \p A, which has a dimension of \f$m \times r\f$.
         */
        LAPACKFullMatrixExt<Number> QA, RA;
        A.reduced_qr(QA, RA);

        /**
         * Perform reduced QR decomposition to component matrix \p B, which
         * has a dimension of \f$n \times r\f$.
         */
        LAPACKFullMatrixExt<Number> QB, RB;
        B.reduced_qr(QB, RB);

        /**
         * Perform SVD to the product \f$R\f$ of the two upper triangular
         * matrices, i.e. \f$R = R_A R_B^T\f$.
         */
        LAPACKFullMatrixExt<Number> R, U_hat, VT_hat;
        /**
         * N.B. Before LAPACK matrix multiplication, the memory of the result
         * matrix should be reinitialized.
         */
        R.reinit(RA.m(), RB.m());
        RA.mTmult(R, RB);
        R.svd(U_hat, Sigma_r, VT_hat);

        U.reinit(QA.m(), formal_rank);
        QA.mmult(U, U_hat);
        VT.reinit(formal_rank, QB.m());
        VT_hat.mTmult(VT, QB);
      }
    else
      {
        is_qr_used = false;

        /**
         * When \p A and \p B are not all long matrices, firstly convert the
         * rank-k matrix to a full matrix, then perform SVD on this full matrix.
         */
        LAPACKFullMatrixExt<Number> fullmatrix(A.m(), B.m());
        A.mTmult(fullmatrix, B);
        fullmatrix.svd(U, Sigma_r, VT);
      }

    if (singular_value_threshold == 0.)
      {
        /**
         * If the singular value threshold is perfect zero, calculate a
         * threshold value instead.
         */
        singular_value_threshold =
          calc_singular_value_threshold(mm, nn, Sigma_r);
      }

    /**
     * Get the actual rank of the matrix by counting the total number of
     * singular values which are larger than the given threshold \p
     * singular_value_threshold. The actual rank should always be less than or
     * equal to \p min_dim.
     */
    size_type rank = 0;
    for (size_t i = 0; i < Sigma_r.size(); i++)
      {
        if (Sigma_r[i] > singular_value_threshold)
          {
            rank++;
          }
      }
    AssertIndexRange(rank, min_dim + 1);

    /**
     * Limit the value of \p truncation_rank not larger than the actual
     * rank just obtained by counting effective singular values.
     */
    if (truncation_rank > rank)
      {
        truncation_rank = rank;
      }

    if (truncation_rank < min_dim)
      {
        if (truncation_rank > 0)
          {
            /**
             * Keep the first \p truncation_rank singular values, while discarding
             * others.
             */
            std::vector<typename numbers::NumberTraits<Number>::real_type> copy(
              std::move(Sigma_r));
            Sigma_r.resize(truncation_rank);
            for (size_type i = 0; i < truncation_rank; i++)
              {
                Sigma_r.at(i) = copy.at(i);
              }

            const size_type Sigma_r_error_size = copy.size() - truncation_rank;
            if (Sigma_r_error_size > 0)
              {
                /**
                 * Copy the remaining singular values for constructing the error
                 * matrices.
                 */
                std::vector<typename numbers::NumberTraits<Number>::real_type>
                  Sigma_r_error(Sigma_r_error_size);
                for (size_type i = 0; i < Sigma_r_error_size; i++)
                  {
                    Sigma_r_error.at(i) = copy.at(i + truncation_rank);
                  }

                /**
                 * Allocate memory for the error matrices. At the moment,
                 * \f$D\f$ stores its transpose.
                 */
                C.reinit(mm, Sigma_r_error_size);
                D.reinit(Sigma_r_error_size, nn);

                C.fill(U, 0, 0, 0, truncation_rank);
                D.fill(VT, 0, 0, truncation_rank, 0);

                if (mm > nn)
                  {
                    /**
                     * When the original full matrix \f$AB^T\f$ is a long
                     * matrix, perform right association.
                     */
                    D.scale_rows(Sigma_r_error);
                    D.transpose();
                  }
                else
                  {
                    /**
                     * When the original full matrix \f$AB^T\f$ is a wide
                     * matrix, perform left association.
                     */
                    C.scale_columns(Sigma_r_error);
                    D.transpose();
                  }

                /**
                 * Balance the Frobenius norm of @p C and @p D.
                 */
                balance_frobenius_norm(C, D);
              }
            else
              {
                C.reinit(0, 0);
                D.reinit(0, 0);
              }

            /**
             * Keep the first \p truncation_rank columns of \p U, while deleting
             * others.
             */
            U.keep_first_n_columns(truncation_rank, true);

            /**
             * Keep the first \p truncation_rank rows of \p VT, while deleting
             * others.
             */
            VT.keep_first_n_rows(truncation_rank, true);
          }
        else
          {
            /**
             * When the truncation rank is zero, the error matrices should be
             * directly copied from the original component matrices.
             */
            C = A;
            D = B;

            /**
             * Balance the Frobenius norm of @p C and @p D.
             */
            balance_frobenius_norm(C, D);

            /**
             * Then, clear the SVD result matrix and singular value vector.
             */
            U.reinit(0, 0);
            VT.reinit(0, 0);
            Sigma_r.clear();
          }
      }
    else
      {
        /**
         * In this case, it could only be that \p truncation_rank == \p min_dim.
         */
        AssertDimension(truncation_rank, min_dim);

        /**
         * There is no effective rank truncation and thus no accuracy loss.
         * Therefore, the error matrix components \f$C\f$ and \f$D\f$ are set to
         * zero dimension.
         */
        C.reinit(0, 0);
        D.reinit(0, 0);

        if (is_qr_used)
          {
            /**
             * In this case, the results \p U, \p Sigma_r and \p VT are obtained
             * via QR decomposition. And if \p M has a dimension \f$m \times n\f$,
             * the dimensions of all matrices are:
             * * \f$U \in \mathbb{R}^{m \times {\rm formal rank}}\f$
             * * \f$\Sigma_r \in \mathbb{R}^{{\rm formal rank} \times {\rm
             * formal rank}}\f$
             * * \f$V \in \mathbb{R}^{n \times {\rm formal rank}}\f$
             */
            if (truncation_rank < formal_rank)
              {
                /**
                 * Keep the first \p truncation_rank singular values, while
                 * discarding others.
                 */
                std::vector<typename numbers::NumberTraits<Number>::real_type>
                  copy(std::move(Sigma_r));
                Sigma_r.resize(truncation_rank);
                for (size_type i = 0; i < truncation_rank; i++)
                  {
                    Sigma_r.at(i) = copy.at(i);
                  }

                /**
                 * Keep the first \p truncation_rank columns of \p U, while
                 * deleting others.
                 */
                U.keep_first_n_columns(truncation_rank, true);

                /**
                 * Keep the first \p truncation_rank rows of \p VT, while deleting
                 * others.
                 */
                VT.keep_first_n_rows(truncation_rank, true);
              }
          }
        else
          {
            /**
             * In this case, the results \p U, \p Sigma_r and \p VT are obtained
             * from full matrix SVD. And if \p M has a dimension \f$m \times
             * n\f$, the dimensions of all matrices obtained from SVD are:
             * * \f$U \in \mathbb{R}^{m \times m}\f$
             * * \f$\Sigma_r \in \mathbb{R}^{m \times n}\f$
             * * \f$V \in \mathbb{R}^{n \times n}\f$
             */
            if (mm > nn)
              {
                /**
                 * When the original matrix is long, \f$\Sigma_r =
                 * \begin{pmatrix}\Sigma_r' \\ 0 \end{pmatrix}\f$. Therefore, we
                 * keep the first \p min_dim columns of \p U, while deleting
                 * others. \p VT is kept intact.
                 */
                U.keep_first_n_columns(min_dim, true);
              }
            else if (mm < nn)
              {
                /**
                 * When the original matrix is wide, \f$\Sigma_r =
                 * \begin{pmatrix} \Sigma_r' & 0 \end{pmatrix}\f$. Therefore, we
                 * keep the first \p
                 * min_dim rows of \p VT, while deleting others. \p U is kept
                 * intact.
                 */
                VT.keep_first_n_rows(min_dim, true);
              }
            else
              {
                /**
                 * When the original matrix is square, we do nothing, since
                 * the truncation rank is the same as the matrix dimension.
                 */
              }
          }
      }

    AssertDimension(U.n(), Sigma_r.size());
    AssertDimension(VT.m(), Sigma_r.size());

    return truncation_rank;
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const size_type size)
    : LAPACKFullMatrix<Number>(size)
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const size_type rows,
                                                   const size_type cols)
    : LAPACKFullMatrix<Number>(rows, cols)
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
    const LAPACKFullMatrixExt &mat)
    : LAPACKFullMatrix<Number>(mat)
    , state(mat.state)
    , property(mat.property)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(property);
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
    const LAPACKFullMatrix<Number> &mat)
    : LAPACKFullMatrix<Number>(mat)
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
    const std::array<types::global_dof_index, 2> &row_index_range,
    const std::array<types::global_dof_index, 2> &column_index_range,
    const LAPACKFullMatrixExt<Number>            &M)
    : LAPACKFullMatrix<Number>(row_index_range[1] - row_index_range[0],
                               column_index_range[1] - column_index_range[0])
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);

    /**
     * Extract the data for the submatrix defined on the block cluster \f$\tau
     * \times \sigma\f$ from the full global matrix \p M.
     */
    for (size_type i = 0; i < LAPACKFullMatrix<Number>::m(); i++)
      {
        for (size_type j = 0; j < LAPACKFullMatrix<Number>::n(); j++)
          {
            /**
             * Because \p M is global, the indices in \f$\tau\f$ and \f$\sigma\f$
             * can be directly used for accessing the elements of \p M.
             */
            (*this)(i, j) =
              M(row_index_range[0] + i, column_index_range[0] + j);
          }
      }
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
    const std::array<types::global_dof_index, 2> &row_index_range,
    const std::array<types::global_dof_index, 2> &column_index_range,
    const LAPACKFullMatrixExt<Number>            &M,
    const std::array<types::global_dof_index, 2> &M_row_index_range,
    const std::array<types::global_dof_index, 2> &M_column_index_range)
    : LAPACKFullMatrix<Number>(row_index_range[1] - row_index_range[0],
                               column_index_range[1] - column_index_range[0])
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Make an assertion that the index ranges for the matrix to be built should
     * be subsets of the index ranges for the local matrix @p M.
     */
    Assert(is_subset(row_index_range, M_row_index_range) &&
             is_subset(column_index_range, M_column_index_range),
           ExcInternalError());

    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);

    /**
     * Extract the data for the submatrix block \f$b = \tau \times \sigma\f$
     * from
     * the original matrix \p M.
     */
    for (size_type i = 0; i < LAPACKFullMatrix<Number>::m(); i++)
      {
        for (size_type j = 0; j < LAPACKFullMatrix<Number>::n(); j++)
          {
            (*this)(i, j) =
              M(row_index_range[0] - M_row_index_range[0] + i,
                column_index_range[0] - M_column_index_range[0] + j);
          }
      }
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
    const LAPACKFullMatrixExt &M1,
    const LAPACKFullMatrixExt &M2,
    bool                       is_horizontal_split)
    : LAPACKFullMatrix<Number>()
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);

    if (is_horizontal_split)
      {
        M1.vstack((*this), M2);
      }
    else
      {
        M1.hstack((*this), M2);
      }
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
    const LAPACKFullMatrixExt &M11,
    const LAPACKFullMatrixExt &M12,
    const LAPACKFullMatrixExt &M21,
    const LAPACKFullMatrixExt &M22)
    : LAPACKFullMatrix<Number>(M11.m() + M21.m(), M11.n() + M12.n())
    , state(LAPACKSupport::matrix)
    , property(LAPACKSupport::general)
    , tau(0)
    , work()
    , iwork()
    , ipiv()
  {
    AssertDimension(M11.m(), M12.m());
    AssertDimension(M21.m(), M22.m());
    AssertDimension(M11.n(), M21.n());
    AssertDimension(M12.n(), M22.n());

    /**
     * Set the property of the parent class as well.
     */
    this->LAPACKFullMatrix<Number>::set_property(LAPACKSupport::general);

    this->fill(M11, 0, 0);
    this->fill(M12, 0, M11.n());
    this->fill(M21, M11.m(), 0);
    this->fill(M22, M11.m(), M11.n());
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number> &
  LAPACKFullMatrixExt<Number>::operator=(
    const LAPACKFullMatrixExt<Number> &matrix)
  {
    LAPACKFullMatrix<Number>::operator=(matrix);
    state                             = matrix.state;
    property                          = matrix.property;
    /**
     * Since \p ipiv contains the crucial permutation data if the matrix has been
     * factorized by LU, it needs to be copied.
     */
    ipiv = matrix.ipiv;

    return (*this);
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number> &
  LAPACKFullMatrixExt<Number>::operator=(const LAPACKFullMatrix<Number> &matrix)
  {
    LAPACKFullMatrix<Number>::operator=(matrix);

    /**
     * Because \p state and \p property are private members of \p
     * LAPACKFullMatrix<Number>, they cannot be copied from the given \p matrix.
     * Hence, they are assigned with the default values \p LAPACKSupport::matrix
     * and \p LAPACKSupport::general.
     */
    state    = LAPACKSupport::matrix;
    property = LAPACKSupport::general;

    return (*this);
  }


  template <typename Number>
  LAPACKFullMatrixExt<Number> &
  LAPACKFullMatrixExt<Number>::operator=(const Number d)
  {
    LAPACKFullMatrix<Number>::operator=(d);

    state = LAPACKSupport::matrix;

    return (*this);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::reinit(const size_type nrows,
                                      const size_type ncols)
  {
    LAPACKFullMatrix<Number>::reinit(nrows, ncols);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::set_column_zeros(const size_type col_index)
  {
    size_type mm = this->m();
    size_type nn = this->n();

    AssertIndexRange(col_index, nn);

    for (size_type i = 0; i < mm; i++)
      {
        this->set(i, col_index, Number());
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::set_row_zeros(const size_type row_index)
  {
    const size_type mm = this->m();
    const size_type nn = this->n();

    AssertIndexRange(row_index, mm);

    for (size_type j = 0; j < nn; j++)
      {
        this->set(row_index, j, Number());
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::get_row(const size_type row_index,
                                       Vector<Number> &row_values) const
  {
    const size_type n_cols = this->n();

    AssertIndexRange(row_index, this->m());

    row_values.reinit(n_cols);

    for (size_type j = 0; j < n_cols; j++)
      {
        row_values(j) = (*this)(row_index, j);
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::get_column(const size_type col_index,
                                          Vector<Number> &col_values) const
  {
    const size_type n_rows = this->m();
    AssertIndexRange(col_index, this->n());

    col_values.reinit(n_rows);

    for (size_type i = 0; i < n_rows; i++)
      {
        col_values(i) = (*this)(i, col_index);
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::remove_row(const size_type row_index)
  {
    AssertIndexRange(row_index, this->m());

    // Number of rows and columns in the resulted matrix.
    const size_type nrows = this->m() - 1;
    const size_type ncols = this->n();

    // Make a shallow copy of the current matrix via rvalue reference by using
    // std::move.
    TransposeTable<Number> copy(std::move(*this));

    // Reinitialize the current matrix with one row deleted (with memory
    // allocation).
    this->TransposeTable<Number>::reinit(nrows, ncols);

    // Perform matrix data copy in row major.
    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            // Ignore the row \p row_index during the copy.
            const size_type ii = (i < row_index ? i : i + 1);
            (*this)(i, j)      = copy(ii, j);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::remove_rows(
    const std::vector<size_type> &row_indices)
  {
    const size_type nrows_for_deletion = row_indices.size();

    // Number of rows and columns in the resulted matrix.
    const size_type nrows_before_deletion = this->m();

    Assert(nrows_for_deletion < nrows_before_deletion && nrows_for_deletion > 0,
           ExcIndexRange(nrows_for_deletion, 1, nrows_before_deletion));

    const size_type nrows = nrows_before_deletion - nrows_for_deletion;
    const size_type ncols = this->n();

    // Make a shallow copy of the current matrix via rvalue reference by using
    // std::move.
    TransposeTable<Number> copy(std::move(*this));

    // Allocate memory and reinitialize the current matrix with several rows
    // deleted.
    this->TransposeTable<Number>::reinit(nrows, ncols);

    // Counter for the row for deletion.
    size_type counter_for_row_deletion = 0;

    // Counter for the row of resulted matrix.
    size_type i = 0;

    // Perform matrix data copy in column major.
    for (size_type ii = 0; ii < nrows_before_deletion; ii++)
      {
        // Ignore the row for deletion in the original matrix.
        if ((counter_for_row_deletion < nrows_for_deletion) &&
            (ii == row_indices.at(counter_for_row_deletion)))
          {
            counter_for_row_deletion++;

            continue;
          }
        else
          {
            // Copy the current row.
            for (size_type j = 0; j < ncols; j++)
              {
                (*this)(i, j) = copy(ii, j);
              }

            i++;
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::keep_first_n_rows(const size_type n,
                                                 bool other_rows_removed)
  {
    AssertIndexRange(n, this->m() + 1);

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    if (n > 0)
      {
        if (other_rows_removed)
          {
            // Make a shallow copy of the current matrix via rvalue reference by
            // using std::move.
            TransposeTable<Number> copy(std::move(*this));
            // Allocate memory and reinitialize the current matrix with several
            // rows deleted.
            this->TransposeTable<Number>::reinit(n, ncols);

            for (size_type j = 0; j < ncols; j++)
              {
                for (size_type i = 0; i < n; i++)
                  {
                    (*this)(i, j) = copy(i, j);
                  }
              }
          }
        else
          {
            for (size_type j = 0; j < ncols; j++)
              {
                for (size_type i = n; i < nrows; i++)
                  {
                    (*this)(i, j) = Number();
                  }
              }
          }
      }
    else
      {
        this->reinit(0, 0);
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::remove_column(const size_type column_index)
  {
    AssertIndexRange(column_index, this->n());

    // Number of rows and columns in the resulted matrix.
    const size_type nrows = this->m();
    const size_type ncols = this->n() - 1;

    // Make a shallow copy of the current matrix via rvalue reference by using
    // std::move.
    TransposeTable<Number> copy(std::move(*this));

    // Reinitialize the current matrix with one column deleted (with memory
    // allocation).
    this->TransposeTable<Number>::reinit(nrows, ncols);

    // Perform matrix data copy in row major.
    for (size_type j = 0; j < ncols; j++)
      {
        // Ignore the column \p column_index during the copy.
        const size_type jj = (j < column_index ? j : j + 1);

        for (size_type i = 0; i < nrows; i++)
          {
            (*this)(i, j) = copy(i, jj);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::remove_columns(
    const std::vector<size_type> &column_indices)
  {
    const size_type ncols_for_deletion = column_indices.size();

    // Number of rows and columns in the resulted matrix.
    const size_type nrows                 = this->m();
    const size_type ncols_before_deletion = this->n();

    Assert(ncols_for_deletion < ncols_before_deletion && ncols_for_deletion > 0,
           ExcIndexRange(ncols_for_deletion, 1, ncols_before_deletion));

    const size_type ncols = ncols_before_deletion - ncols_for_deletion;

    // Make a shallow copy of the current matrix via rvalue reference by using
    // std::move.
    TransposeTable<Number> copy(std::move(*this));

    // Allocate memory and reinitialize the current matrix with several columns
    // deleted.
    this->TransposeTable<Number>::reinit(nrows, ncols);

    // Counter for the column for deletion.
    size_type counter_for_col_deletion = 0;


    // Counter for the column of resulted matrix.
    size_type j = 0;
    // Perform matrix data copy in row major.
    for (size_type jj = 0; jj < ncols_before_deletion; jj++)
      {
        // Ignore the column for deletion in the original matrix.
        if ((counter_for_col_deletion < ncols_for_deletion) &&
            (jj == column_indices.at(counter_for_col_deletion)))
          {
            counter_for_col_deletion++;

            continue;
          }
        else
          {
            // Copy the current column.
            for (size_type i = 0; i < nrows; i++)
              {
                (*this)(i, j) = copy(i, jj);
              }

            j++;
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::keep_first_n_columns(const size_type n,
                                                    bool other_columns_removed)
  {
    AssertIndexRange(n, this->n() + 1);

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    if (n > 0)
      {
        if (other_columns_removed)
          {
            // Make a shallow copy of the current matrix via rvalue reference by
            // using std::move.
            TransposeTable<Number> copy(std::move(*this));
            // Allocate memory and reinitialize the current matrix with several
            // columns deleted.
            this->TransposeTable<Number>::reinit(nrows, n);

            for (size_type j = 0; j < n; j++)
              {
                for (size_type i = 0; i < nrows; i++)
                  {
                    (*this)(i, j) = copy(i, j);
                  }
              }
          }
        else
          {
            for (size_type j = n; j < ncols; j++)
              {
                for (size_type i = 0; i < nrows; i++)
                  {
                    (*this)(i, j) = Number();
                  }
              }
          }
      }
    else
      {
        this->reinit(0, 0);
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::svd(
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT)
  {
    Assert(state == LAPACKSupport::matrix ||
             state == LAPACKSupport::State::cholesky,
           LAPACKSupport::ExcState(state));
    state = LAPACKSupport::unusable;

    const size_type mm      = this->m();
    const size_type nn      = this->n();
    const size_type min_dim = std::min(mm, nn);

    /**
     * Allocate memory for result matrices.
     */
    U.reinit(mm, mm);
    VT.reinit(nn, nn);
    Sigma_r.resize(min_dim);
    std::fill(Sigma_r.begin(), Sigma_r.end(), 0.);

    /**
     * Integer array as the work space for pivoting, which is the \p IWORK
     * parameter in \p dgesdd.
     */
    ipiv.resize(8 * mm);

    /**
     * Return status.
     */
    types::blas_int info = 0;

    /**
     * Determine the optimal work space size, which will be stored into \p
     * real_work. At the moment, \p work is a scalar which will hold the queried
     * optimal \p lwork. Its first element should be initialized as zero.
     */
    std::vector<Number> work(1);
    work[0] = Number();
    types::blas_int                                                lwork = -1;
    std::vector<typename numbers::NumberTraits<Number>::real_type> real_work;
    if (numbers::NumberTraits<Number>::is_complex)
      {
        // This array is only used by the complex versions.
        std::size_t min = std::min(this->m(), this->n());
        std::size_t max = std::max(this->m(), this->n());
        real_work.resize(std::max(5 * min * min + 5 * min,
                                  2 * max * min + 2 * min * min + min));
      }

    LAPACKHelpers::gesdd_helper(
      LAPACKSupport::A,
      mm,
      nn,
      this->values,
      Sigma_r,
      U.values,
      VT.values,
      work, //! Store the optimal work space size after inquiry.
      real_work,
      ipiv,  //! Integer work space for pivoting.
      lwork, //! lwork = -1, for an inquiry of the optimal work space size.
      info);

    AssertThrow(info == 0, LAPACKSupport::ExcErrorCode("gesdd", info));
    /**
     * Update the work space size and resize the \p work vector.
     */
    lwork = static_cast<types::blas_int>(std::abs(work[0]) + 1);
    work.resize(lwork);

    /**
     * Perform the real SVD.
     */
    LAPACKHelpers::gesdd_helper(LAPACKSupport::A,
                                mm,
                                nn,
                                this->values,
                                Sigma_r,
                                U.values,
                                VT.values,
                                work,
                                real_work,
                                ipiv, //! Integer work space for pivoting.
                                lwork,
                                info);

    state = LAPACKSupport::svd;
  }

  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::svd(
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT,
    const size_type truncation_rank)
  {
    const size_type mm      = this->m();
    const size_type nn      = this->n();
    const size_type min_dim = std::min(mm, nn);

    /**
     * Perform the full SVD. After the operation,
     */
    svd(U, Sigma_r, VT);

    /**
     * Perform singular value truncation when the specified rank is less than
     * the minimum dimension.
     */
    if (truncation_rank < min_dim)
      {
        /**
         * Keep the first \p truncation_rank number of singular values and clear
         * the remaining ones.
         */
        for (size_type i = truncation_rank; i < Sigma_r.size(); i++)
          {
            Sigma_r.at(i) = Number();
          }

        /**
         * Keep the first \p truncation_rank columns of \p U, while setting other
         * columns to zero.
         */
        U.keep_first_n_columns(truncation_rank, false);

        /**
         * Keep the first \p truncation_rank rows of \p VT, while setting other
         * rows to zero.
         */
        VT.keep_first_n_rows(truncation_rank, false);
      }
    else
      {
        /**
         * Do not thing when the truncation rank is equal to or larger than \p
         * min_dim. This means the truncation is an identity operator.
         */
      }
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::reduced_svd(
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT,
    size_type truncation_rank,
    Number    singular_value_threshold)
  {
    const size_type mm      = this->m();
    const size_type nn      = this->n();
    const size_type min_dim = std::min(mm, nn);

    /**
     * <dl class="section">
     *   <dt>Work flow</dt>
     *   <dd>
     */

    /**
     * Perform the full SVD.
     */
    svd(U, Sigma_r, VT);

    if (singular_value_threshold == 0.)
      {
        /**
         * If the singular value threshold is perfect zero, calculate a
         * threshold value instead.
         */
        singular_value_threshold =
          calc_singular_value_threshold(mm, nn, Sigma_r);
      }

    /**
     * Get the actual rank of the matrix by counting the total number of
     * singular values which are larger than the given threshold \p
     * singular_value_threshold. The actual rank should always be less than or
     * equal to \p min_dim.
     */
    size_type rank = 0;
    for (size_t i = 0; i < Sigma_r.size(); i++)
      {
        if (Sigma_r[i] > singular_value_threshold)
          {
            rank++;
          }
      }
    AssertIndexRange(rank, min_dim + 1);

    /**
     * Limit the value of \p truncation_rank not larger than the actual
     * rank just obtained by counting effective singular values.
     */
    if (truncation_rank > rank)
      {
        truncation_rank = rank;
      }

    if (truncation_rank < min_dim)
      {
        if (truncation_rank > 0)
          {
            /**
             * Perform singular value truncation when the given \p truncation_rank
             * (after value limiting wrt. the actual rank) is less than the
             * minimum matrix dimension. The procedures are as below.
             *
             * 1. Keep the first \p truncation_rank singular values, while discarding
             * the others.
             */
            std::vector<typename numbers::NumberTraits<Number>::real_type> copy(
              std::move(Sigma_r));
            Sigma_r.resize(truncation_rank);
            for (size_type i = 0; i < truncation_rank; i++)
              {
                Sigma_r.at(i) = copy.at(i);
              }

            /**
             * 2. Keep the first \p truncation_rank columns of \p U, while discarding
             * the others.
             */
            U.keep_first_n_columns(truncation_rank, true);

            /**
             * 3. Keep the first \p truncation_rank rows of \p VT, while discarding
             * the others.
             */
            VT.keep_first_n_rows(truncation_rank, true);
          }
        else
          {
            /**
             * Clear the SVD result matrix and singular value vector.
             */
            U.reinit(0, 0);
            VT.reinit(0, 0);
            Sigma_r.clear();
          }
      }
    else
      {
        /**
         * When \p truncation_rank (after value limiting wrt. the actual rank) is
         * equal to the minimum matrix dimension, we only need to adjust the
         * columns of \f$U\f$ or the rows of \f$V^T\f$ depending on the shape of
         * the matrix \f$M\f$.
         */
        AssertDimension(truncation_rank, min_dim);

        /**
         * For details, if \f$M\f$ has a dimension \f$m \times n\f$, the
         * dimensions of all matrices obtained from SVD are:
         * * \f$U \in \mathbb{R}^{m \times m}\f$
         * * \f$\Sigma_r \in \mathbb{R}^{m \times n}\f$
         * * \f$V \in \mathbb{R}^{n \times n}\f$
         */
        if (mm > nn)
          {
            /**
             * When \f$M\f$ is long, \f$\Sigma_r =
             * \begin{pmatrix}\Sigma_r' \\ 0 \end{pmatrix}\f$. Therefore, we
             * keep
             * the first \p min_dim columns of \f$U\f$, while deleting others.
             * \f$V^T\f$ is kept intact.
             */
            U.keep_first_n_columns(min_dim, true);
          }
        else if (mm < nn)
          {
            /**
             * When \f$M\f$ is wide, \f$\Sigma_r = \begin{pmatrix}
             * \Sigma_r' & 0 \end{pmatrix}\f$. Therefore, we keep the first \p
             * min_dim rows of \f$V^T\f$, while deleting others. \f$U\f$ is kept
             * intact.
             */
            VT.keep_first_n_rows(min_dim, true);
          }
        else
          {
            /**
             * When \f$M\f$ is square, do nothing.
             */
          }
      }

    AssertDimension(U.n(), Sigma_r.size());
    AssertDimension(VT.m(), Sigma_r.size());

    /**
     * Return the value of \p truncation_rank. Instead of its original specified
     * value, it now contains the actual rank of the matrix after truncation.
     */
    return truncation_rank;

    /**
     *   </dd>
     * </dl>
     */
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::reduced_svd(
    LAPACKFullMatrixExt<Number>                                    &U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number>                                    &VT,
    LAPACKFullMatrixExt<Number>                                    &C,
    LAPACKFullMatrixExt<Number>                                    &D,
    size_type truncation_rank,
    Number    singular_value_threshold)
  {
    const size_type mm      = this->m();
    const size_type nn      = this->n();
    const size_type min_dim = std::min(mm, nn);

    /**
     * <dl class="section">
     *   <dt>Work flow</dt>
     *   <dd>
     */

    /**
     * Perform the full SVD.
     */
    svd(U, Sigma_r, VT);

    if (singular_value_threshold == 0.)
      {
        /**
         * If the singular value threshold is perfect zero, calculate a
         * threshold value instead.
         */
        singular_value_threshold =
          calc_singular_value_threshold(mm, nn, Sigma_r);
      }

    /**
     * Get the actual rank of the matrix by counting the total number of
     * singular values which are larger than the given threshold \p
     * singular_value_threshold. The actual rank should always be less than or
     * equal to \p min_dim.
     */
    size_type rank = 0;
    for (size_t i = 0; i < Sigma_r.size(); i++)
      {
        if (Sigma_r[i] > singular_value_threshold)
          {
            rank++;
          }
      }
    AssertIndexRange(rank, min_dim + 1);

    /**
     * Limit the value of \p truncation_rank not larger than the actual
     * rank just obtained by counting effective singular values.
     */
    if (truncation_rank > rank)
      {
        truncation_rank = rank;
      }

    if (truncation_rank < min_dim)
      {
        if (truncation_rank > 0)
          {
            /**
             * Perform singular value truncation when the given \p truncation_rank
             * (after value limiting wrt. the actual rank) is less than the
             * minimum matrix dimension. The procedures are as below.
             *
             * 1. Keep the first \p truncation_rank singular values, while discarding
             * the others.
             */
            std::vector<typename numbers::NumberTraits<Number>::real_type> copy(
              std::move(Sigma_r));
            Sigma_r.resize(truncation_rank);
            for (size_type i = 0; i < truncation_rank; i++)
              {
                Sigma_r.at(i) = copy.at(i);
              }

            const size_type Sigma_r_error_size = copy.size() - truncation_rank;
            if (Sigma_r_error_size > 0)
              {
                /**
                 * Copy the remaining singular values for constructing the error
                 * matrices.
                 */
                std::vector<typename numbers::NumberTraits<Number>::real_type>
                  Sigma_r_error(Sigma_r_error_size);
                for (size_type i = 0; i < Sigma_r_error_size; i++)
                  {
                    Sigma_r_error.at(i) = copy.at(i + truncation_rank);
                  }

                /**
                 * Allocate memory for the error matrices. At the moment,
                 * \f$D\f$ stores its transpose.
                 */
                C.reinit(mm, Sigma_r_error_size);
                D.reinit(Sigma_r_error_size, nn);

                C.fill(U, 0, 0, 0, truncation_rank);
                D.fill(VT, 0, 0, truncation_rank, 0);

                if (mm > nn)
                  {
                    /**
                     * When the original full matrix \f$AB^T\f$ is a long
                     * matrix, perform right association.
                     */
                    D.scale_rows(Sigma_r_error);
                    D.transpose();
                  }
                else
                  {
                    /**
                     * When the original full matrix \f$AB^T\f$ is a wide
                     * matrix, perform left association.
                     */
                    C.scale_columns(Sigma_r_error);
                    D.transpose();
                  }

                /**
                 * Balance the Frobenius norm of @p C and @p D.
                 */
                balance_frobenius_norm(C, D);
              }
            else
              {
                C.reinit(0, 0);
                D.reinit(0, 0);
              }

            /**
             * 2. Keep the first \p truncation_rank columns of \p U, while discarding
             * the others.
             */
            U.keep_first_n_columns(truncation_rank, true);

            /**
             * 3. Keep the first \p truncation_rank rows of \p VT, while discarding
             * the others.
             */
            VT.keep_first_n_rows(truncation_rank, true);
          }
        else
          {
            /**
             * When the truncation rank is zero, the error matrices should be
             * directly obtained from the SVD results.
             */
            C = U;
            D = VT;

            if (mm > nn)
              {
                /**
                 * When the original full matrix \f$AB^T\f$ is a long matrix,
                 * perform right association.
                 */
                D.scale_rows(Sigma_r);
                D.transpose();
              }
            else
              {
                /**
                 * When the original full matrix \f$AB^T\f$ is a wide matrix,
                 * perform left association.
                 */
                C.scale_columns(Sigma_r);
                D.transpose();
              }

            /**
             * Balance the Frobenius norm of @p C and @p D.
             */
            balance_frobenius_norm(C, D);

            /**
             * Then, clear the SVD result matrix and singular value vector.
             */
            U.reinit(0, 0);
            VT.reinit(0, 0);
            Sigma_r.clear();
          }
      }
    else
      {
        /**
         * When \p truncation_rank (after value limiting wrt. the actual rank) is
         * equal to the minimum matrix dimension, we only need to adjust the
         * columns of \f$U\f$ or the rows of \f$V^T\f$ depending on the shape of
         * the matrix \f$M\f$.
         */
        AssertDimension(truncation_rank, min_dim);

        /**
         * There is no effective rank truncation and thus no accuracy loss.
         * Therefore, the error matrix components \f$C\f$ and \f$D\f$ are set to
         * zero dimension.
         */
        C.reinit(0, 0);
        D.reinit(0, 0);

        /**
         * For details, if \f$M\f$ has a dimension \f$m \times n\f$, the
         * dimensions of all matrices obtained from SVD are:
         * * \f$U \in \mathbb{R}^{m \times m}\f$
         * * \f$\Sigma_r \in \mathbb{R}^{m \times n}\f$
         * * \f$V \in \mathbb{R}^{n \times n}\f$
         */
        if (mm > nn)
          {
            /**
             * When \f$M\f$ is long, \f$\Sigma_r =
             * \begin{pmatrix}\Sigma_r' \\ 0 \end{pmatrix}\f$. Therefore, we
             * keep
             * the first \p min_dim columns of \f$U\f$, while deleting others.
             * \f$V^T\f$ is kept intact.
             */
            U.keep_first_n_columns(min_dim, true);
          }
        else if (mm < nn)
          {
            /**
             * When \f$M\f$ is wide, \f$\Sigma_r = \begin{pmatrix}
             * \Sigma_r' & 0 \end{pmatrix}\f$. Therefore, we keep the first \p
             * min_dim rows of \f$V^T\f$, while deleting others. \f$U\f$ is kept
             * intact.
             */
            VT.keep_first_n_rows(min_dim, true);
          }
        else
          {
            /**
             * When \f$M\f$ is square, do nothing.
             */
          }
      }

    AssertDimension(U.n(), Sigma_r.size());
    AssertDimension(VT.m(), Sigma_r.size());

    /**
     * Return the value of \p truncation_rank. Instead of its original specified
     * value, it now contains the actual rank of the matrix after truncation.
     */
    return truncation_rank;

    /**
     *   </dd>
     * </dl>
     */
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::compute_lu_factorization()
  {
    Assert(state == matrix, ExcState(state));
    state = LAPACKSupport::unusable;

    const types::blas_int mm     = this->m();
    const types::blas_int nn     = this->n();
    Number *const         values = this->values.data();
    ipiv.resize(mm);
    types::blas_int info = 0;
    getrf(&mm, &nn, values, &mm, ipiv.data(), &info);

    Assert(info >= 0, ExcInternalError());

    // if info >= 0, the factorization has been completed
    state = LAPACKSupport::lu;

    AssertThrow(info == 0, LACExceptions::ExcSingular());
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::compute_cholesky_factorization()
  {
    Assert(state == matrix, ExcState(state));
    Assert(property == LAPACKSupport::symmetric, ExcProperty(property));
    state = LAPACKSupport::unusable;

    /**
     * Call the member function in the parent class so that the matrix property
     * in the parent class can be property set.
     */
    LAPACKFullMatrix<Number>::compute_cholesky_factorization();

    state    = LAPACKSupport::cholesky;
    property = LAPACKSupport::lower_triangular;
  }


  template <typename Number>
  std::vector<types::blas_int> &
  LAPACKFullMatrixExt<Number>::get_lu_permutation()
  {
    return ipiv;
  }


  template <typename Number>
  const std::vector<types::blas_int> &
  LAPACKFullMatrixExt<Number>::get_lu_permutation() const
  {
    return ipiv;
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::rank(Number threshold) const
  {
    size_type rank;

    if (this->m() != 0 && this->n() != 0)
      {
        LAPACKFullMatrixExt<Number> copy(*this);
        LAPACKFullMatrixExt<Number> U, VT;
        std::vector<typename numbers::NumberTraits<Number>::real_type> Sigma_r;

        copy.svd(U, Sigma_r, VT);

        if (threshold == 0.)
          {
            /**
             * If the singular value threshold is perfect zero, calculate a
             * threshold value instead.
             */
            threshold =
              calc_singular_value_threshold(this->m(), this->n(), Sigma_r);
          }

        rank = 0;
        for (size_t i = 0; i < Sigma_r.size(); i++)
          {
            if (Sigma_r[i] > threshold)
              {
                rank++;
              }
          }
      }
    else
      {
        rank = 0;
      }

    return rank;
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::qr(LAPACKFullMatrixExt<Number> &Q,
                                  LAPACKFullMatrixExt<Number> &R)
  {
    const types::blas_int mm = this->m();
    const types::blas_int nn = this->n();

    tau.resize(std::min(mm, nn));
    std::fill(tau.begin(), tau.end(), 0.);

    /**
     * Set work space size as 1 and \p lwork as -1 for the determination of
     * optimal work space size.
     */
    work.resize(1);
    types::blas_int lwork = -1;

    /**
     * Make sure that the first entry in the work space is clear, in case the
     * routine does not completely overwrite the memory:
     */
    types::blas_int info = 0;
    work[0]              = Number();

    LAPACKHelpers::geqrf_helper(mm, nn, this->values, tau, work, lwork, info);
    AssertThrow(info == 0, LAPACKSupport::ExcErrorCode("geqrf", info));

    /**
     * Resize the work space and add one to the size computed by LAPACK to be on
     * the safe side.
     */
    lwork = static_cast<types::blas_int>(std::abs(work[0]) + 1);
    work.resize(lwork);

    /**
     * Perform the actual QR decomposition.
     */
    LAPACKHelpers::geqrf_helper(mm, nn, this->values, tau, work, lwork, info);
    AssertThrow(info == 0, LAPACKSupport::ExcErrorCode("geqrf", info));

    /**
     * Collect results for the upper triangular matrix \p R.
     */
    R.reinit(mm, nn);

    for (types::blas_int i = 0; i < mm; i++)
      {
        for (types::blas_int j = i; j < nn; j++)
          {
            R(i, j) = (*this)(i, j);
          }
      }

    /**
     * Collect results for the orthogonal matrix \p Q with a dimension \f$m \times
     * m\f$. It is represented as a product of elementary reflectors
     * (Householder transformation) as \f[ Q = H_1 H_2 \cdots H_k, \f] where
     * \f$k = \min\{m, n\}\f$.
     */
    Q.reinit(mm, mm);
    LAPACKFullMatrixExt<Number> Q_work(mm, mm);
    LAPACKFullMatrixExt<Number> H(mm, mm);

    for (types::blas_int i = 0; i < std::min(mm, nn); i++)
      {
        /**
         * Construct the vector \p v. Values in \p v before the i'th component are
         * all zeros. The i'th component is 1. Values after the i'th component
         * are
         * stored in the current matrix \p A(i+1:m,i).
         */
        Vector<Number> v(mm);
        v(i) = 1.;
        for (types::blas_int j = i + 1; j < mm; j++)
          {
            v(j) = (*this)(j, i);
          }

        /**
         * Construct the Householder matrix.
         */
        for (types::blas_int j = 0; j < mm; j++)
          {
            for (types::blas_int k = 0; k < mm; k++)
              {
                H(j, k) = ((j == k) ? 1.0 : 0.) - tau[i] * v(j) * v(k);
              }
          }

        if (i == 0)
          {
            Q      = H;
            Q_work = H;
          }
        else
          {
            Q_work.mmult(Q, H);
            Q_work = Q;
          }
      }

    /**
     * Release the work space used.
     */
    work.resize(0);
    tau.resize(0);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::reduced_qr(LAPACKFullMatrixExt<Number> &Q,
                                          LAPACKFullMatrixExt<Number> &R)
  {
    /**
     * Perform the standard QR decomposition.
     */
    qr(Q, R);

    if (this->m() > this->n())
      {
        /**
         * Perform the reduced QR decomposition by keeping the first \p n columns
         * of Q and the first \p n rows of R.
         */
        Q.keep_first_n_columns(this->n());
        R.keep_first_n_rows(this->n());
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::scale_rows(
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
  {
    Assert(state == LAPACKSupport::matrix ||
             state == LAPACKSupport::inverse_matrix,
           ExcState(state));
    AssertDimension(this->m(), V.size());

    size_type nrows = this->m();
    size_type ncols = this->n();

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            (*this)(i, j) *= V.at(i);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::scale_rows(
    LAPACKFullMatrixExt<Number>                                          &A,
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
    const
  {
    AssertDimension(this->m(), V.size());

    size_type nrows = this->m();
    size_type ncols = this->n();

    A.reinit(nrows, ncols);

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            A(i, j) = (*this)(i, j) * V.at(i);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::scale_columns(
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
  {
    Assert(state == LAPACKSupport::matrix ||
             state == LAPACKSupport::inverse_matrix,
           ExcState(state));
    AssertDimension(this->n(), V.size());

    size_type nrows = this->m();
    size_type ncols = this->n();

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            (*this)(i, j) *= V.at(j);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::scale_columns(
    LAPACKFullMatrixExt<Number>                                          &A,
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
    const
  {
    AssertDimension(this->n(), V.size());

    size_type nrows = this->m();
    size_type ncols = this->n();

    A.reinit(nrows, ncols);

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            A(i, j) = (*this)(i, j) * V.at(j);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::scale_columns(
    const Vector<typename numbers::NumberTraits<Number>::real_type> &V)
  {
    Assert(state == LAPACKSupport::matrix ||
             state == LAPACKSupport::inverse_matrix,
           ExcState(state));
    AssertDimension(this->n(), V.size());

    size_type nrows = this->m();
    size_type ncols = this->n();

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            (*this)(i, j) *= V(j);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::transpose()
  {
    Assert(state == LAPACKSupport::matrix ||
             state == LAPACKSupport::inverse_matrix,
           ExcState(state));

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    // Make a shallow copy of the current matrix via rvalue reference by using
    // std::move.
    TransposeTable<Number> copy(std::move(*this));

    // Reinitialize the current matrix with memory allocation.
    this->TransposeTable<Number>::reinit(ncols, nrows);

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            (*this)(j, i) = copy(i, j);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::transpose(LAPACKFullMatrixExt<Number> &AT) const
  {
    const size_type nrows = this->m();
    const size_type ncols = this->n();

    AT.reinit(ncols, nrows);

    for (size_type j = 0; j < ncols; j++)
      {
        for (size_type i = 0; i < nrows; i++)
          {
            AT(j, i) = (*this)(i, j);
          }
      }
  }


  template <typename Number>
  template <typename MatrixType>
  void
  LAPACKFullMatrixExt<Number>::fill(const MatrixType &src,
                                    const size_type   dst_offset_i,
                                    const size_type   dst_offset_j,
                                    const size_type   src_offset_i,
                                    const size_type   src_offset_j,
                                    const Number      factor,
                                    const bool        transpose,
                                    const bool        is_adding)
  {
    AssertIndexRange(src_offset_i, src.m());
    AssertIndexRange(src_offset_j, src.n());
    AssertIndexRange(dst_offset_i, this->m());
    AssertIndexRange(dst_offset_j, this->n());

    /**
     * Determine the number of rows and columns to be copied.
     */
    size_type nrows_for_copy, ncols_for_copy;

    if (transpose)
      {
        nrows_for_copy =
          std::min(src.n() - src_offset_j, this->m() - dst_offset_i);
        ncols_for_copy =
          std::min(src.m() - src_offset_i, this->n() - dst_offset_j);
      }
    else
      {
        nrows_for_copy =
          std::min(src.m() - src_offset_i, this->m() - dst_offset_i);
        ncols_for_copy =
          std::min(src.n() - src_offset_j, this->n() - dst_offset_j);
      }

    if (transpose)
      {
        for (size_type i = 0; i < nrows_for_copy; i++)
          {
            for (size_type j = 0; j < ncols_for_copy; j++)
              {
                if (is_adding)
                  {
                    (*this)(dst_offset_i + i, dst_offset_j + j) +=
                      factor * src(src_offset_i + j, src_offset_j + i);
                  }
                else
                  {
                    (*this)(dst_offset_i + i, dst_offset_j + j) =
                      factor * src(src_offset_i + j, src_offset_j + i);
                  }
              }
          }
      }
    else
      {
        for (size_type i = 0; i < nrows_for_copy; i++)
          {
            for (size_type j = 0; j < ncols_for_copy; j++)
              {
                if (is_adding)
                  {
                    (*this)(dst_offset_i + i, dst_offset_j + j) +=
                      factor * src(src_offset_i + i, src_offset_j + j);
                  }
                else
                  {
                    (*this)(dst_offset_i + i, dst_offset_j + j) =
                      factor * src(src_offset_i + i, src_offset_j + j);
                  }
              }
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::fill_row(const size_type       row_index,
                                        const Vector<Number> &values,
                                        const bool            is_adding)
  {
    const size_type n_cols = this->n();

    AssertIndexRange(row_index, this->m());
    AssertDimension(values.size(), n_cols);

    for (size_type j = 0; j < n_cols; j++)
      {
        if (is_adding)
          {
            (*this)(row_index, j) += values(j);
          }
        else
          {
            (*this)(row_index, j) = values(j);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::fill_row(const size_type dst_row_index,
                                        const size_type src_row_index,
                                        const LAPACKFullMatrixExt<Number> &M,
                                        const Number factor,
                                        const bool   is_adding)
  {
    const size_type n_cols = this->n();

    AssertIndexRange(dst_row_index, this->m());
    AssertIndexRange(src_row_index, M.m());
    AssertDimension(M.n(), n_cols);

    for (size_type j = 0; j < n_cols; j++)
      {
        if (is_adding)
          {
            (*this)(dst_row_index, j) += M(src_row_index, j) * factor;
          }
        else
          {
            (*this)(dst_row_index, j) = M(src_row_index, j) * factor;
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::fill_rows(
    const std::array<types::global_dof_index, 2> &row_index_range,
    const LAPACKFullMatrixExt<Number>            &M,
    const std::array<types::global_dof_index, 2> &M_row_index_range,
    const Number                                  factor,
    const bool                                    is_adding)
  {
    Assert(is_subset(M_row_index_range, row_index_range), ExcInternalError());
    AssertDimension(this->n(), M.n());

    for (size_type i = 0; i < M.m(); i++)
      {
        this->fill_row(M_row_index_range[0] - row_index_range[0] + i,
                       i,
                       M,
                       factor,
                       is_adding);
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::fill_col(const size_type       col_index,
                                        const Vector<Number> &values,
                                        const bool            is_adding)
  {
    const size_type n_rows = this->m();
    AssertIndexRange(col_index, this->n());
    AssertDimension(values.size(), n_rows);

    for (size_type i = 0; i < n_rows; i++)
      {
        if (is_adding)
          {
            (*this)(i, col_index) += values(i);
          }
        else
          {
            (*this)(i, col_index) = values(i);
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::hstack(
    LAPACKFullMatrixExt<Number>       &C,
    const LAPACKFullMatrixExt<Number> &B) const
  {
    AssertDimension(this->m(), B.m());

    const size_type nrows = this->m();
    const size_type ncols = this->n() + B.n();

    C.reinit(nrows, ncols);

    C.fill((*this), 0, 0, 0, 0);
    C.fill(B, 0, this->n(), 0, 0);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::vstack(
    LAPACKFullMatrixExt<Number>       &C,
    const LAPACKFullMatrixExt<Number> &B) const
  {
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m() + B.m();
    const size_type ncols = this->n();

    C.reinit(nrows, ncols);

    C.fill((*this), 0, 0, 0, 0);
    C.fill(B, this->m(), 0, 0, 0);
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::rank_k_decompose(const unsigned int           k,
                                                LAPACKFullMatrixExt<Number> &A,
                                                LAPACKFullMatrixExt<Number> &B)
  {
    if (this->n() < this->m())
      {
        /**
         * When the matrix is long, apply right associativity.
         */
        return rank_k_decompose(k, A, B, false);
      }
    else
      {
        /**
         * When the matrix is wide, apply left associativity.
         */
        return rank_k_decompose(k, A, B, true);
      }
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::rank_k_decompose(const unsigned int           k,
                                                LAPACKFullMatrixExt<Number> &A,
                                                LAPACKFullMatrixExt<Number> &B,
                                                LAPACKFullMatrixExt<Number> &C,
                                                LAPACKFullMatrixExt<Number> &D)
  {
    if (this->n() < this->m())
      {
        /**
         * When the matrix is long, apply right associativity.
         */
        return rank_k_decompose(k, A, B, C, D, false);
      }
    else
      {
        /**
         * When the matrix is wide, apply left associativity.
         */
        return rank_k_decompose(k, A, B, C, D, true);
      }
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::rank_k_decompose(LAPACKFullMatrixExt<Number> &A,
                                                LAPACKFullMatrixExt<Number> &B)
  {
    /**
     * Use the minimum matrix dimension as the default truncation rank.
     */
    const unsigned int k = std::min(this->m(), this->n());

    if (this->n() < this->m())
      {
        return rank_k_decompose(k, A, B, false);
      }
    else
      {
        return rank_k_decompose(k, A, B, true);
      }
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::rank_k_decompose(const unsigned int           k,
                                                LAPACKFullMatrixExt<Number> &A,
                                                LAPACKFullMatrixExt<Number> &B,
                                                bool is_left_associative)
  {
    std::vector<typename numbers::NumberTraits<Number>::real_type> Sigma_r;

    /**
     * Perform RSVD for the matrix and return U and VT into A and B
     * respectively. N.B. After running this function, B actually holds the
     * transposition of itself at the moment.
     */
    const size_type effective_rank = this->reduced_svd(A, Sigma_r, B, k);

    if (effective_rank > 0)
      {
        if (is_left_associative)
          {
            /**
             * Let A = A*Sigma_r and B = B^T.
             */
            A.scale_columns(Sigma_r);
            B.transpose();
          }
        else
          {
            /**
             * Let A = A and B = (Sigma_r * B)^T.
             */
            B.scale_rows(Sigma_r);
            B.transpose();
          }
      }

    return effective_rank;
  }


  template <typename Number>
  typename LAPACKFullMatrixExt<Number>::size_type
  LAPACKFullMatrixExt<Number>::rank_k_decompose(const unsigned int           k,
                                                LAPACKFullMatrixExt<Number> &A,
                                                LAPACKFullMatrixExt<Number> &B,
                                                LAPACKFullMatrixExt<Number> &C,
                                                LAPACKFullMatrixExt<Number> &D,
                                                bool is_left_associative)
  {
    std::vector<typename numbers::NumberTraits<Number>::real_type> Sigma_r;

    /**
     * Perform RSVD for the matrix and return U and VT into A and B
     * respectively. N.B. After running this function, B actually holds the
     * transposition of itself at the moment.
     */
    const size_type effective_rank = this->reduced_svd(A, Sigma_r, B, C, D, k);

    if (effective_rank > 0)
      {
        if (is_left_associative)
          {
            /**
             * Let A = A*Sigma_r and B = B^T.
             */
            A.scale_columns(Sigma_r);
            B.transpose();
          }
        else
          {
            /**
             * Let A = A and B = (Sigma_r * B)^T.
             */
            B.scale_rows(Sigma_r);
            B.transpose();
          }
      }

    return effective_rank;
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::add(
    LAPACKFullMatrixExt<Number>       &C,
    const LAPACKFullMatrixExt<Number> &B,
    const bool                         is_result_matrix_symm_apriori) const
  {
    AssertDimension(this->m(), B.m());
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    C.reinit(nrows, ncols);

    if (is_result_matrix_symm_apriori)
      {
        /**
         * Perform addition for matrix elements in the diagonal part and in the
         * lower triangular part only.
         */
        Assert(C.get_property() == LAPACKSupport::symmetric,
               ExcMessage(std::string("The result matrix should be ") +
                          std::string(LAPACKSupport::property_name(
                            LAPACKSupport::symmetric))));

        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j <= i; j++)
              {
                C(i, j) = (*this)(i, j) + B(i, j);
              }
          }
      }
    else
      {
        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j < ncols; j++)
              {
                C(i, j) = (*this)(i, j) + B(i, j);
              }
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::add(
    LAPACKFullMatrixExt<Number>       &C,
    const Number                       b,
    const LAPACKFullMatrixExt<Number> &B,
    const bool                         is_result_matrix_symm_apriori) const
  {
    AssertDimension(this->m(), B.m());
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    C.reinit(nrows, ncols);

    if (is_result_matrix_symm_apriori)
      {
        /**
         * Perform addition for matrix elements in the diagonal part and in the
         * lower triangular part only.
         */
        Assert(C.get_property() == LAPACKSupport::symmetric,
               ExcMessage(std::string("The result matrix should be ") +
                          std::string(LAPACKSupport::property_name(
                            LAPACKSupport::symmetric))));

        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j <= i; j++)
              {
                C(i, j) = (*this)(i, j) + b * B(i, j);
              }
          }
      }
    else
      {
        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j < ncols; j++)
              {
                C(i, j) = (*this)(i, j) + b * B(i, j);
              }
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::add(const LAPACKFullMatrixExt<Number> &B,
                                   const bool is_result_matrix_store_tril_only)
  {
    AssertDimension(this->m(), B.m());
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    if (is_result_matrix_store_tril_only)
      {
        /**
         * Perform addition for matrix elements in the diagonal part and in the
         * lower triangular part only.
         */
        Assert(this->get_property() == LAPACKSupport::symmetric ||
                 this->get_property() == LAPACKSupport::lower_triangular,
               ExcInvalidLAPACKFullMatrixProperty(this->get_property()));

        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j <= i; j++)
              {
                (*this)(i, j) += B(i, j);
              }
          }
      }
    else
      {
        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j < ncols; j++)
              {
                (*this)(i, j) += B(i, j);
              }
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::add(const Number                       b,
                                   const LAPACKFullMatrixExt<Number> &B,
                                   const bool is_result_matrix_store_tril_only)
  {
    AssertDimension(this->m(), B.m());
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m();
    const size_type ncols = this->n();

    if (is_result_matrix_store_tril_only)
      {
        /**
         * Perform addition for matrix elements in the diagonal part and in the
         * lower triangular part only.
         */
        Assert(this->get_property() == LAPACKSupport::symmetric ||
                 this->get_property() == LAPACKSupport::lower_triangular,
               ExcInvalidLAPACKFullMatrixProperty(this->get_property()));

        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j <= i; j++)
              {
                (*this)(i, j) += b * B(i, j);
              }
          }
      }
    else
      {
        for (size_type i = 0; i < nrows; i++)
          {
            for (size_type j = 0; j < ncols; j++)
              {
                (*this)(i, j) += b * B(i, j);
              }
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::vmult(Vector<Number>       &w,
                                     const Vector<Number> &v,
                                     const bool            adding) const
  {
    switch (get_property())
      {
          case LAPACKSupport::symmetric: {
            LAPACKHelpers::symv_helper(LAPACKSupport::L,
                                       1.0,
                                       this->m(),
                                       this->values,
                                       v.data(),
                                       (adding ? 1.0 : 0.0),
                                       w.data());

            break;
          }
          case LAPACKSupport::general: {
            /**
             * Call the normal matrix-vector multiplication member function of
             * the parent class.
             */
            this->LAPACKFullMatrix<Number>::vmult(w, v, adding);

            break;
          }
          case LAPACKSupport::Property::lower_triangular: {
            w = v;
            LAPACKHelpers::trmv_helper(LAPACKSupport::L,
                                       LAPACKSupport::N,
                                       LAPACKSupport::N,
                                       this->m(),
                                       this->values,
                                       w.data());

            break;
          }
          case LAPACKSupport::Property::upper_triangular: {
            w = v;
            LAPACKHelpers::trmv_helper(LAPACKSupport::U,
                                       LAPACKSupport::N,
                                       LAPACKSupport::N,
                                       this->m(),
                                       this->values,
                                       w.data());

            break;
          }
          default: {
            Assert(false, ExcNotImplemented());

            break;
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::Tvmult(Vector<Number>       &w,
                                      const Vector<Number> &v,
                                      const bool            adding) const
  {
    switch (get_property())
      {
          case LAPACKSupport::symmetric: {
            this->vmult(w, v, adding);

            break;
          }
          case LAPACKSupport::general: {
            this->LAPACKFullMatrix<Number>::Tvmult(w, v, adding);

            break;
          }
          case LAPACKSupport::Property::lower_triangular: {
            w = v;
            LAPACKHelpers::trmv_helper(LAPACKSupport::L,
                                       LAPACKSupport::T,
                                       LAPACKSupport::N,
                                       this->m(),
                                       this->values,
                                       w.data());

            break;
          }
          case LAPACKSupport::Property::upper_triangular: {
            w = v;
            LAPACKHelpers::trmv_helper(LAPACKSupport::U,
                                       LAPACKSupport::T,
                                       LAPACKSupport::N,
                                       this->m(),
                                       this->values,
                                       w.data());

            break;
          }
          default: {
            Assert(false, ExcNotImplemented());

            break;
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::mmult(LAPACKFullMatrixExt<Number>       &C,
                                     const LAPACKFullMatrixExt<Number> &B,
                                     const bool adding) const
  {
    AssertDimension(this->n(), B.m());

    const size_type nrows = this->m();
    const size_type ncols = B.n();

    if (C.m() != nrows || C.n() != ncols)
      {
        C.reinit(nrows, ncols);
      }

    // Call the \p mmult function in the parent class which operates on \p
    // LAPACKFullMatrix<Number>.
    LAPACKFullMatrix<Number>::mmult(C, (LAPACKFullMatrix<Number>)B, adding);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::mmult(LAPACKFullMatrixExt<Number>       &C,
                                     const Number                       alpha,
                                     const LAPACKFullMatrixExt<Number> &B,
                                     const bool adding) const
  {
    AssertDimension(this->n(), B.m());

    const size_type nrows = this->m();
    const size_type ncols = B.n();

    if (C.m() != nrows || C.n() != ncols)
      {
        C.reinit(nrows, ncols);
      }

    /**
     * Make a local copy of the matrix \p B and scale it.
     */
    LAPACKFullMatrixExt<Number> B_scaled(B);
    B_scaled *= alpha;

    // Call the \p mmult function in the parent class which operates on \p
    // LAPACKFullMatrix<Number>.
    LAPACKFullMatrix<Number>::mmult(C,
                                    (LAPACKFullMatrix<Number>)B_scaled,
                                    adding);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::mTmult(LAPACKFullMatrixExt<Number>       &C,
                                      const LAPACKFullMatrixExt<Number> &B,
                                      const bool adding) const
  {
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m();
    const size_type ncols = B.m();

    if (C.m() != nrows || C.n() != ncols)
      {
        C.reinit(nrows, ncols);
      }

    // Call the \p mTmult function in the parent class which operates on \p
    // LAPACKFullMatrix<Number>.
    LAPACKFullMatrix<Number>::mTmult(C, (LAPACKFullMatrix<Number>)B, adding);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::mTmult(LAPACKFullMatrixExt<Number>       &C,
                                      const Number                       alpha,
                                      const LAPACKFullMatrixExt<Number> &B,
                                      const bool adding) const
  {
    AssertDimension(this->n(), B.n());

    const size_type nrows = this->m();
    const size_type ncols = B.m();

    if (C.m() != nrows || C.n() != ncols)
      {
        C.reinit(nrows, ncols);
      }

    /**
     * Make a local copy of the matrix \p B and scale it.
     */
    LAPACKFullMatrixExt<Number> B_scaled(B);
    B_scaled *= alpha;

    // Call the \p mTmult function in the parent class which operates on \p
    // LAPACKFullMatrix<Number>.
    LAPACKFullMatrix<Number>::mTmult(C,
                                     (LAPACKFullMatrix<Number>)B_scaled,
                                     adding);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::Tmmult(LAPACKFullMatrixExt<Number>       &C,
                                      const LAPACKFullMatrixExt<Number> &B,
                                      const bool adding) const
  {
    AssertDimension(this->m(), B.m());

    const size_type nrows = this->n();
    const size_type ncols = B.n();

    if (C.m() != nrows || C.n() != ncols)
      {
        C.reinit(nrows, ncols);
      }

    // Call the \p Tmmult function in the parent class which operates on \p
    // LAPACKFullMatrix<Number>.
    LAPACKFullMatrix<Number>::Tmmult(C, (LAPACKFullMatrix<Number>)B, adding);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::Tmmult(LAPACKFullMatrixExt<Number>       &C,
                                      const Number                       alpha,
                                      const LAPACKFullMatrixExt<Number> &B,
                                      const bool adding) const
  {
    AssertDimension(this->m(), B.m());

    const size_type nrows = this->n();
    const size_type ncols = B.n();

    if (C.m() != nrows || C.n() != ncols)
      {
        C.reinit(nrows, ncols);
      }

    /**
     * Make a local copy of the matrix \p B and scale it.
     */
    LAPACKFullMatrixExt<Number> B_scaled(B);
    B_scaled *= alpha;

    // Call the \p Tmmult function in the parent class which operates on \p
    // LAPACKFullMatrix<Number>.
    LAPACKFullMatrix<Number>::Tmmult(C,
                                     (LAPACKFullMatrix<Number>)B_scaled,
                                     adding);
  }


  template <typename Number>
  Number
  LAPACKFullMatrixExt<Number>::determinant2x2() const
  {
    AssertDimension(this->m(), this->n());
    AssertDimension(this->m(), 2);

    return (*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0);
  }


  template <typename Number>
  Number
  LAPACKFullMatrixExt<Number>::determinant3x3() const
  {
    AssertDimension(this->m(), this->n());
    AssertDimension(this->m(), 3);

    return (*this)(0, 0) * (*this)(1, 1) * (*this)(2, 2) +
           (*this)(0, 1) * (*this)(1, 2) * (*this)(2, 0) +
           (*this)(0, 2) * (*this)(1, 0) * (*this)(2, 1) -
           (*this)(0, 2) * (*this)(1, 1) * (*this)(2, 0) -
           (*this)(0, 1) * (*this)(1, 0) * (*this)(2, 2) -
           (*this)(0, 0) * (*this)(1, 2) * (*this)(2, 1);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::invert(const LAPACKFullMatrixExt<Number> &M)
  {
    AssertDimension(M.m(), M.n());

    (*this) = M;
    this->LAPACKFullMatrix<Number>::invert();
    this->state = LAPACKSupport::State::inverse_matrix;
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::invert_by_gauss_elim(
    LAPACKFullMatrixExt<Number> &M_inv)
  {
    AssertDimension(this->m(), this->n());

    const size_type n = this->m();

    M_inv.reinit(n, n);

    /**
     * Eliminate the lower triangular part of the matrix.
     */
    for (size_type l = 0; l < n; l++)
      {
        /**
         * Scale the current row by the factor \f$\frac{1}{a_{ll}}\f$ and this
         * value is directly filled into the result matrix.
         */
        M_inv(l, l) = 1.0 / (*this)(l, l);

        /**
         * For the matrix \f$M\f$, after this scaling, the element \f$M_{ll}\f$
         * will become 1, there is no need to compute the scaling of this
         * element. Meanwhile, the elements \f$M_{l1}, \cdots, M_{l,l-1}\f$ are
         * already zeros after previous eliminations, neither need we compute
         * the scaling of these elements. Therefore, the actual computation to
         * be performed is for columns \f$j = l + 1, \cdots, n\f$.
         */
        for (size_type j = l + 1; j < n; j++)
          {
            (*this)(l, j) = M_inv(l, l) * (*this)(l, j);
          }

        /**
         * For the matrix \f$M^{-1}\f$, after this scaling, the element at
         * \f$(l, l)\f$ will be \f$\frac{1}{M_{ll}}\f$, so there is no need to
         * compute this scaling. Meanwhile, only those elements \f$M^{-1}_{l,1},
         * \cdots, M^{-1}_{l-1}\f$ may be non-zeros, we only loop over \f$j = 1,
         * \cdots, l
         * - 1\f$.
         */
        for (size_type j = 0; j < l; j++)
          {
            M_inv(l, j) = M_inv(l, l) * M_inv(l, j);
          }

        /**
         * Then we eliminate the elements \f$M_{l+1,l}, \cdots, M_{n,l}\f$ by
         * iterating over the rows \f$l + 1, \cdots, n\f$.
         */
        for (size_type i = l + 1; i < n; i++)
          {
            /**
             * This transformation will only influence the columns \f$1, \cdots,
             * l\f$ in \f$M^{-1}\f$.
             */
            for (size_type j = 0; j <= l; j++)
              {
                M_inv(i, j) = M_inv(i, j) - (*this)(i, l) * M_inv(l, j);
              }

            /**
             * This transformation will only influence the columns \f$l+1,
             * \cdots, n\f$ in \f$M\f$.
             */
            for (size_type j = l + 1; j < n; j++)
              {
                (*this)(i, j) = (*this)(i, j) - (*this)(i, l) * (*this)(l, j);
              }
          }
      }

    /**
     * Eliminate the upper triangular part. Now the row transformation is only
     * related to the result matrix \f$M^{-1}\f$.
     */
    for (size_type l = n - 1; l > 0; l--)
      {
        /**
         * Eliminate the elements \f$M_{l-1,l}, \cdots, M_{1,l}\f$.
         */
        size_type i = l - 1;
        /**
         * N.B. When the loop counter \p i of \p unsigned type decreased to be
         * zero, further decrement will not produce a value smaller than zero
         * but
         * a very large integer. Hence, we do not use a typical \p for loop here
         * as below.
         *
         * <code> for (size_type i = l - 1; i >= 0; i--)
         * {
         *   ...
         * }
         * </code>
         *
         * Instead, we use a \p while loop with the \p true condition. Inside this
         * loop, when we detect the counter is zero after loop execution, we
         * jump out the loop.
         */
        while (true)
          {
            /**
             * Because the elements in the current l'th row of \f$M^{-1}\f$ are
             * generally non-zeros, the computation involves columns \f$1,
             * \cdots, n\f$.
             */
            for (size_type j = 0; j < n; j++)
              {
                M_inv(i, j) = M_inv(i, j) - (*this)(i, l) * M_inv(l, j);
              }

            if (i == 0)
              {
                break;
              }
            else
              {
                i--;
              }
          }
      }
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::set_property(
    const LAPACKSupport::Property property)
  {
    this->property = property;
    /**
     * Set the matrix property in the parent class to be the same.
     */
    this->LAPACKFullMatrix<Number>::set_property(property);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::set_state(const LAPACKSupport::State state)
  {
    this->state = state;
    /**
     * N.B. There is no public or protected member function provided by
     * @p LAPACKFullMatrix for setting its matrix state. Hence, unlike in the
     * member function @p set_property, we do nothing here to the parent class.
     */
  }


  template <typename Number>
  LAPACKSupport::Property
  LAPACKFullMatrixExt<Number>::get_property() const
  {
    return property;
  }

  template <typename Number>
  LAPACKSupport::State
  LAPACKFullMatrixExt<Number>::get_state() const
  {
    return state;
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::solve(Vector<Number> &v,
                                     const bool      transposed) const
  {
    Assert(this->m() == this->n(), LACExceptions::ExcNotQuadratic());
    AssertDimension(this->m(), v.size());

    const char           *trans  = transposed ? &T : &N;
    const types::blas_int nn     = this->n();
    const Number *const   values = this->values.data();
    const types::blas_int n_rhs  = 1;
    types::blas_int       info   = 0;

    if (state == State::lu)
      {
        getrs(
          trans, &nn, &n_rhs, values, &nn, ipiv.data(), v.begin(), &nn, &info);
      }
    else if (state == State::cholesky)
      {
        potrs(
          &LAPACKSupport::L, &nn, &n_rhs, values, &nn, v.begin(), &nn, &info);
      }
    else if (property == Property::upper_triangular ||
             property == Property::lower_triangular)
      {
        const char uplo =
          (property == upper_triangular ? LAPACKSupport::U : LAPACKSupport::L);

        const types::blas_int lda = nn;
        const types::blas_int ldb = nn;
        trtrs(
          &uplo, trans, "N", &nn, &n_rhs, values, &lda, v.begin(), &ldb, &info);
      }
    else
      {
        Assert(false,
               ExcMessage(
                 "The matrix has to be either factorized or triangular."));
      }

    Assert(info == 0, ExcInternalError());
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::solve_by_forward_substitution(
    Vector<Number> &b,
    const bool      transposed,
    const bool      is_unit_diagonal,
    const bool      permute_rhs_vector) const
  {
    // The matrix should be square.
    AssertDimension(this->m(), this->n());
    AssertDimension(this->n(), b.size());

    char uplo;

    if (transposed)
      {
        uplo = 'U';
      }
    else
      {
        uplo = 'L';
      }

    /**
     * Permute RHS vector if the current matrix is obtained from LU
     * factorization.
     */
    if (state == LAPACKSupport::State::lu && permute_rhs_vector)
      {
        // @p ipiv is the vector storing the permutations applied for pivoting in
        // the LU factorization. Hence, we make an assertion about its size.
        Assert(ipiv.size() > 0, ExcInternalError());

        permute_vector_by_ipiv(b, ipiv);
      }

    /**
     * \alert{The member variable \p values of \p b is private, which cannot be
     * directly accessed. Therefore, the function \p trsv_helper is designed to
     * accept the data pointer of \p b, which can be obtained via \p b.data().}
     */
    LAPACKHelpers::trsv_helper(
      uplo, transposed, is_unit_diagonal, this->m(), this->values, b.data());
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::solve_by_forward_substitution(
    Vector<Number>       &x,
    const Vector<Number> &b,
    const bool            transposed,
    const bool            is_unit_diagonal) const
  {
    x = b;
    solve_by_forward_substitution(x, transposed, is_unit_diagonal);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::solve_by_backward_substitution(
    Vector<Number> &b,
    const bool      transposed,
    const bool      is_unit_diagonal) const
  {
    // The matrix should be square.
    AssertDimension(this->m(), this->n());

    char uplo;

    if (transposed)
      {
        uplo = 'L';
      }
    else
      {
        uplo = 'U';
      }

    /**
     * \alert{The member variable \p values of \p b is private, which cannot be
     * directly accessed. Therefore, the function \p trsv_helper is designed to
     * accept the data pointer of \p b, which can be obtained via \p b.data().}
     */
    LAPACKHelpers::trsv_helper(
      uplo, transposed, is_unit_diagonal, this->m(), this->values, b.data());
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::solve_by_backward_substitution(
    Vector<Number>       &x,
    const Vector<Number> &b,
    const bool            transposed,
    const bool            is_unit_diagonal) const
  {
    x = b;
    solve_by_backward_substitution(x, transposed, is_unit_diagonal);
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::print_formatted_to_mat(
    std::ostream      &out,
    const std::string &name,
    const unsigned int precision,
    const bool         scientific,
    const unsigned int width,
    const char        *zero_string,
    const double       denominator,
    const double       threshold) const
  {
    out << "# name: " << name << "\n";
    out << "# type: matrix\n";
    out << "# rows: " << this->m() << "\n";
    out << "# columns: " << this->n() << "\n";

    this->print_formatted(
      out, precision, scientific, width, zero_string, denominator, threshold);

    out << "\n\n";
  }


  template <typename Number>
  void
  LAPACKFullMatrixExt<Number>::read_from_mat(std::ifstream     &in,
                                             const std::string &name)
  {
    std::string line_buf;

    /**
     * Iterate over each line of the file to search the desired variable.
     */
    while (std::getline(in, line_buf))
      {
        if (line_buf.compare(std::string("# name: ") + name) == 0)
          {
            /**
             * When the desired variable is found, read the next line to check
             * the
             * data type is \p matrix.
             */
            std::getline(in, line_buf);
            Assert(line_buf.compare("# type: matrix") == 0,
                   ExcMessage(
                     "Data type for the matrix to be read should be 'matrix'"));

            /**
             * Read a new line to extract the number of rows.
             */
            std::getline(in, line_buf);
            std::smatch sm;
            bool        found =
              std::regex_match(line_buf, sm, std::regex("# rows: (\\d+)"));
            Assert(found, ExcMessage("Cannot get n_rows of the matrix!"));
            MAYBE_UNUSED(found);
            const unsigned int n_rows = std::stoi(sm.str(1));

            /**
             * Read a new line to extract the number of columns.
             */
            std::getline(in, line_buf);
            found =
              std::regex_match(line_buf, sm, std::regex("# columns: (\\d+)"));
            Assert(found, ExcMessage("Cannot get n_cols of the matrix!"));
            const unsigned int n_cols = std::stoi(sm.str(1));

            Assert(n_rows > 0, ExcMessage("Matrix to be read has no rows!"));
            Assert(n_cols > 0, ExcMessage("Matrix to be read has no columns!"));

            reinit(n_rows, n_cols);
            /**
             * Get each row of the matrix.
             */
            for (size_type i = 0; i < n_rows; i++)
              {
                std::getline(in, line_buf);
                std::istringstream line_buf_stream(line_buf);
                /**
                 * Get each matrix element in a row.
                 */
                for (size_type j = 0; j < n_cols; j++)
                  {
                    line_buf_stream >> (*this)(i, j);
                  }
              }

            /**
             * After reading all matrix data, exit from the loop.
             */
            break;
          }
      }
  }


  template <typename Number>
  std::size_t
  LAPACKFullMatrixExt<Number>::memory_consumption() const
  {
    return TransposeTable<Number>::memory_consumption() +
           this->ipiv.capacity() * sizeof(types::blas_int) +
           this->iwork.capacity() * sizeof(types::blas_int) +
           this->work.capacity() * sizeof(Number) +
           this->tau.capacity() *
             sizeof(typename numbers::NumberTraits<Number>::real_type) +
           sizeof(LAPACKSupport::State) + sizeof(LAPACKSupport::Property);
  }


  template <typename Number>
  std::size_t
  LAPACKFullMatrixExt<Number>::memory_consumption_for_core_data() const
  {
    return this->m() * this->n() * sizeof(Number);
  }
} // namespace HierBEM

#endif /* INCLUDE_LAPACK_FULL_MATRIX_EXT_H_ */
