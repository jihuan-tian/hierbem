/**
 * \file lapack_linalg.h
 * \brief Linear algebra computation using LAPACK functions.
 * \date 2021-06-10
 * \author Jihuan Tian
 */

#ifndef INCLUDE_LAPACK_FULL_MATRIX_EXT_H_
#define INCLUDE_LAPACK_FULL_MATRIX_EXT_H_

#include "general_exceptions.h"
#include "lapack_helpers.h"

using namespace dealii;

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
          const std::vector<Number> &  values,
          LAPACKFullMatrixExt<Number> &matrix);

  /**
   * Perform SVD on the product of two component matrices \f$A\f$ and \f$B^T\f$.
   * @param A
   * @param B
   * @param U
   * @param Sigma_r
   * @param VT
   */
  static size_type
  reduced_svd_on_AxBT(
    LAPACKFullMatrixExt<Number> &                                   A,
    LAPACKFullMatrixExt<Number> &                                   B,
    LAPACKFullMatrixExt<Number> &                                   U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number> &                                   VT);

  /**
   * Perform SVD on the product of two component matrices \f$A\f$ and \f$B^T\f$
   * with rank truncation.
   * @param A
   * @param B
   * @param U
   * @param Sigma_r
   * @param VT
   * @param truncation_rank
   */
  static size_type
  reduced_svd_on_AxBT(
    LAPACKFullMatrixExt<Number> &                                   A,
    LAPACKFullMatrixExt<Number> &                                   B,
    LAPACKFullMatrixExt<Number> &                                   U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number> &                                   VT,
    const size_type truncation_rank);

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
   * @param mat
   */
  LAPACKFullMatrixExt(const LAPACKFullMatrix<Number> &mat);

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
   * @param A
   * @param U
   * @param Sigma_r the list of singular values, which has a dimension of \f$\min(m,n)\f$.
   * @param VT
   */
  void
  svd(LAPACKFullMatrixExt<Number> &                                   U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number> &                                   VT);

  /**
   * Perform the standard singular value decomposition (SVD) with rank
   * truncation.
   * @param A
   * @param U
   * @param Sigma_r the list of singular values, which has a dimension of \f$\min(m,n)\f$.
   * @param VT
   */
  void
  svd(LAPACKFullMatrixExt<Number> &                                   U,
      std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
      LAPACKFullMatrixExt<Number> &                                   VT,
      const size_type truncation_rank);

  /**
   * Perform the reduced singular value decomposition (SVD) with rank
   * truncation.
   * @param A
   * @param U
   * @param Sigma_r
   * @param VT
   * @return Effective rank.
   */
  size_type
  reduced_svd(
    LAPACKFullMatrixExt<Number> &                                   U,
    std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
    LAPACKFullMatrixExt<Number> &                                   VT,
    const size_type truncation_rank);

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
   * Left multiply the matrix with a diagonal matrix \p V, which is stored in a
   * std::vector.
   *
   * @param V
   */
  void
  scale_rows(
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V);

  /**
   * Left multiply the matrix with a diagonal matrix \p V, which is stored in a
   * std::vector. The results are stored in a new matrix.
   * @param A
   * @param V
   */
  void
  scale_rows(
    LAPACKFullMatrixExt<Number> &                                         A,
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
    const;

  /**
   * Right multiply the matrix with a diagonal matrix \p V, which is stored in a
   * std::vector.
   * @param V
   */
  void
  scale_columns(
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V);

  /**
   * Right multiply the matrix with a diagonal matrix \p V, which is stored in a
   * std::vector. The results are stored in a new matrix.
   * @param A
   * @param V
   */
  void
  scale_columns(
    LAPACKFullMatrixExt<Number> &                                         A,
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V)
    const;

  /**
   * Right multiply the matrix with a diagonal matrix \p V, which is stored in a
   * Vector.
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
   * Get the tranpose of the current matrix into a new matrix \p AT.
   * @param AT
   */
  void
  transpose(LAPACKFullMatrixExt<Number> &AT) const;

  /**
   * Fill a rectangular block. This function has the similar behavior as
   * FullMatrix::fill.
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
       const bool        transpose    = false);

  /**
   * Horizontally stack two matrices.
   * @param C
   * @param B
   */
  void
  hstack(LAPACKFullMatrixExt<Number> &      C,
         const LAPACKFullMatrixExt<Number> &B) const;

  /**
   * Vertically stack two matrices.
   * @param C
   * @param B
   */
  void
  vstack(LAPACKFullMatrixExt<Number> &      C,
         const LAPACKFullMatrixExt<Number> &B) const;

  /**
   * Decompose the full matrix into the two components of its rank-k
   * representation, the associativity of triple-matrix multiplication is
   * automatically detected.
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
   * representation.
   * @param k
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
   * Matrix addition \f$C = A + B\f$, where \f$A\f$ is the current matrix.
   */
  void
  add(LAPACKFullMatrixExt<Number> &      C,
      const LAPACKFullMatrixExt<Number> &B) const;

  /**
   * Print a LAPACKFullMatrixExt to Octave mat format.
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
  print_formatted_to_mat(std::ostream &     out,
                         const std::string &name,
                         const unsigned int precision   = 8,
                         const bool         scientific  = true,
                         const unsigned int width       = 0,
                         const char *       zero_string = "0",
                         const double       denominator = 1.,
                         const double       threshold   = 0.) const;

private:
  LAPACKSupport::State state;

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
};


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
LAPACKFullMatrixExt<Number>::ConstantMatrix(const size_type              rows,
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
                                     const std::vector<Number> &  values,
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
  LAPACKFullMatrixExt<Number> &                                   A,
  LAPACKFullMatrixExt<Number> &                                   B,
  LAPACKFullMatrixExt<Number> &                                   U,
  std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
  LAPACKFullMatrixExt<Number> &                                   VT)
{
  AssertDimension(A.n(), B.n());

  const size_type representation_rank = A.n();

  if (A.m() > representation_rank && B.m() > representation_rank)
    {
      /**
       * Perform reduced QR decomposition to component matrix \p A, which
       * has a dimension of \f$m \times r\f$.
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

      U.reinit(QA.m(), representation_rank);
      QA.mmult(U, U_hat);
      VT.reinit(representation_rank, QB.m());
      VT_hat.mTmult(VT, QB);
    }
  else
    {
      /**
       * Firstly convert the rank-k matrix to a full matrix, then perform
       * SVD on this full matrix.
       */
      LAPACKFullMatrixExt<Number> fullmatrix(A.m(), B.m());
      A.mTmult(fullmatrix, B);
      fullmatrix.svd(U, Sigma_r, VT);
    }

  return Sigma_r.size();
}


template <typename Number>
typename LAPACKFullMatrixExt<Number>::size_type
LAPACKFullMatrixExt<Number>::reduced_svd_on_AxBT(
  LAPACKFullMatrixExt<Number> &                                   A,
  LAPACKFullMatrixExt<Number> &                                   B,
  LAPACKFullMatrixExt<Number> &                                   U,
  std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
  LAPACKFullMatrixExt<Number> &                                   VT,
  const size_type truncation_rank)
{
  const size_type effective_rank = reduced_svd_on_AxBT(A, B, U, Sigma_r, VT);

  if (truncation_rank < effective_rank)
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
       * Keep the first \p truncation_rank columns of U.
       */
      U.keep_first_n_columns(truncation_rank);

      /**
       * Keep the first \p truncation_rank rows of VT.
       */
      VT.keep_first_n_rows(truncation_rank);
    }
  else
    {
      /**
       * Do nothing.
       */
    }

  return Sigma_r.size();
}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const size_type size)
  : LAPACKFullMatrix<Number>(size)
  , state(LAPACKSupport::matrix)
  , tau(0)
  , work()
  , iwork()
{}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const size_type rows,
                                                 const size_type cols)
  : LAPACKFullMatrix<Number>(rows, cols)
  , state(LAPACKSupport::matrix)
  , tau(0)
  , work()
  , iwork()
{}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const LAPACKFullMatrixExt &mat)
  : LAPACKFullMatrix<Number>(mat)
  , state(LAPACKSupport::matrix)
  , tau(0)
  , work()
  , iwork()
{}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
  const LAPACKFullMatrix<Number> &mat)
  : LAPACKFullMatrix<Number>(mat)
  , state(LAPACKSupport::matrix)
  , tau(0)
  , work()
  , iwork()
{}


template <typename Number>
void
LAPACKFullMatrixExt<Number>::set_column_zeros(const size_type col_index)
{
  size_type mm = this->m();
  size_type nn = this->n();

  Assert(col_index >= 0 && col_index < nn,
         ExcRightOpenIntervalRange(col_index, 0, nn));

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

  Assert(row_index >= 0 && row_index < mm,
         ExcRightOpenIntervalRange(row_index, 0, mm));

  for (size_type j = 0; j < nn; j++)
    {
      this->set(row_index, j, Number());
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
      /**
       * Do nothing.
       */
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
      /**
       * Do nothing.
       */
    }
}


template <typename Number>
void
LAPACKFullMatrixExt<Number>::svd(
  LAPACKFullMatrixExt<Number> &                                   U,
  std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
  LAPACKFullMatrixExt<Number> &                                   VT)
{
  Assert(state == LAPACKSupport::matrix, LAPACKSupport::ExcState(state));
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
  std::vector<types::blas_int> ipiv(8 * mm);

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
      real_work.resize(
        std::max(5 * min * min + 5 * min, 2 * max * min + 2 * min * min + min));
    }

  internal::LAPACKFullMatrixImplementation::gesdd_helper(
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
  internal::LAPACKFullMatrixImplementation::gesdd_helper(
    LAPACKSupport::A,
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
  LAPACKFullMatrixExt<Number> &                                   U,
  std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
  LAPACKFullMatrixExt<Number> &                                   VT,
  const size_type truncation_rank)
{
  const size_type mm      = this->m();
  const size_type nn      = this->n();
  const size_type min_dim = std::min(mm, nn);

  /**
   * Perform the full SVD.
   */
  svd(U, Sigma_r, VT);

  /**
   * Perform singular value truncation when the specified rank is less than the
   * minimum dimension.
   */
  if (truncation_rank < min_dim)
    {
      /**
       * Clear singular values.
       */
      for (size_type i = truncation_rank; i < Sigma_r.size(); i++)
        {
          Sigma_r.at(i) = Number();
        }

      /**
       * Keep the first \p truncation_rank columns of \p U, while setting others
       * to zero.
       */
      U.keep_first_n_columns(truncation_rank, false);

      /**
       * Keep the first \p truncation_rank rows of \p VT, while setting others
       * to zero.
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
  LAPACKFullMatrixExt<Number> &                                   U,
  std::vector<typename numbers::NumberTraits<Number>::real_type> &Sigma_r,
  LAPACKFullMatrixExt<Number> &                                   VT,
  const size_type truncation_rank)
{
  const size_type mm      = this->m();
  const size_type nn      = this->n();
  const size_type min_dim = std::min(mm, nn);

  /**
   * Perform the full SVD.
   */
  svd(U, Sigma_r, VT);

  /**
   * Perform singular value truncation when the specified rank is less than the
   * minimum dimension.
   */
  if (truncation_rank < min_dim)
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
       * Keep the first \p truncation_rank rows of \p VT, while deleting others.
       */
      VT.keep_first_n_rows(truncation_rank, true);
    }
  else
    {
      if (mm > nn)
        {
          /**
           * Keep the first \p min_dim columns of \p U, while deleting
           * others.
           */
          U.keep_first_n_columns(min_dim, true);
        }
      else if (mm < nn)
        {
          /**
           * Keep the first \p min_dim rows of \p VT, while deleting
           * others.
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

  AssertDimension(U.n(), Sigma_r.size());
  AssertDimension(VT.m(), Sigma_r.size());

  return Sigma_r.size();
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

  internal::LAPACKFullMatrixImplementation::geqrf_helper(
    mm, nn, this->values, tau, work, lwork, info);
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
  internal::LAPACKFullMatrixImplementation::geqrf_helper(
    mm, nn, this->values, tau, work, lwork, info);
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
   * m\f$. It is represented as a product of elementary reflectors (Householder
   * transformation) as
   * \f[
   * Q = H_1 H_2 \cdots H_k,
   * \f]
   * where \f$k = \min\{m, n\}\f$.
   */
  Q.reinit(mm, mm);
  LAPACKFullMatrixExt<Number> Q_work(mm, mm);
  LAPACKFullMatrixExt<Number> H(mm, mm);

  for (types::blas_int i = 0; i < std::min(mm, nn); i++)
    {
      /**
       * Construct the vector \p v. Values in \p v before the i'th component are
       * all zeros. The i'th component is 1. Values after the i'th component are
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
  LAPACKFullMatrixExt<Number> &                                         A,
  const std::vector<typename numbers::NumberTraits<Number>::real_type> &V) const
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
  LAPACKFullMatrixExt<Number> &                                         A,
  const std::vector<typename numbers::NumberTraits<Number>::real_type> &V) const
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
                                  const bool        transpose)
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
              (*this)(dst_offset_i + i, dst_offset_j + j) =
                factor * src(src_offset_i + j, src_offset_j + i);
            }
        }
    }
  else
    {
      for (size_type i = 0; i < nrows_for_copy; i++)
        {
          for (size_type j = 0; j < ncols_for_copy; j++)
            {
              (*this)(dst_offset_i + i, dst_offset_j + j) =
                factor * src(src_offset_i + i, src_offset_j + j);
            }
        }
    }
}


template <typename Number>
void
LAPACKFullMatrixExt<Number>::hstack(LAPACKFullMatrixExt<Number> &      C,
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
LAPACKFullMatrixExt<Number>::vstack(LAPACKFullMatrixExt<Number> &      C,
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
   * Perform RSVD for the matrix and return U and VT into A and B respectively.
   * N.B. After running this function, B actually holdes the tranpose of itself.
   */
  const size_type effective_rank = this->reduced_svd(A, Sigma_r, B, k);

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

  return effective_rank;
}


template <typename Number>
void
LAPACKFullMatrixExt<Number>::add(LAPACKFullMatrixExt<Number> &      C,
                                 const LAPACKFullMatrixExt<Number> &B) const
{
  AssertDimension(this->m(), B.m());
  AssertDimension(this->n(), B.n());

  const size_type nrows = this->m();
  const size_type ncols = this->n();

  C.reinit(nrows, ncols);

  for (size_type i = 0; i < nrows; i++)
    {
      for (size_type j = 0; j < ncols; j++)
        {
          C(i, j) = (*this)(i, j) + B(i, j);
        }
    }
}


template <typename Number>
void
LAPACKFullMatrixExt<Number>::print_formatted_to_mat(
  std::ostream &     out,
  const std::string &name,
  const unsigned int precision,
  const bool         scientific,
  const unsigned int width,
  const char *       zero_string,
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


#endif /* INCLUDE_LAPACK_FULL_MATRIX_EXT_H_ */
