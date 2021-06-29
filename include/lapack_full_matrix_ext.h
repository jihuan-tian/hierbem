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
   * Reshape a vector of values into a LAPACKFullMatrixExt in column major.
   */
  static void
  Reshape(const size_type              rows,
          const size_type              cols,
          const std::vector<Number> &  values,
          LAPACKFullMatrixExt<Number> &matrix);

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
   * Left multiply the matrix with a diagonal matrix \p V, which is stored in a
   * std::vector.
   *
   * @param V
   */
  void
  scale_rows(
    const std::vector<typename numbers::NumberTraits<Number>::real_type> &V);


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
                   bool                         is_left_associative = true);

private:
  LAPACKSupport::State state;
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
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const size_type size)
  : LAPACKFullMatrix<Number>(size)
  , state(LAPACKSupport::matrix)
{}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const size_type rows,
                                                 const size_type cols)
  : LAPACKFullMatrix<Number>(rows, cols)
  , state(LAPACKSupport::matrix)
{}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(const LAPACKFullMatrixExt &mat)
  : LAPACKFullMatrix<Number>(mat)
  , state(LAPACKSupport::matrix)
{}


template <typename Number>
LAPACKFullMatrixExt<Number>::LAPACKFullMatrixExt(
  const LAPACKFullMatrix<Number> &mat)
  : LAPACKFullMatrix<Number>(mat)
  , state(LAPACKSupport::matrix)
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
  AssertIndexRange(n, this->m());

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
  AssertIndexRange(n, this->n());

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
typename LAPACKFullMatrixExt<Number>::size_type
LAPACKFullMatrixExt<Number>::rank_k_decompose(const unsigned int           k,
                                              LAPACKFullMatrixExt<Number> &A,
                                              LAPACKFullMatrixExt<Number> &B,
                                              bool is_left_associative)
{
  std::vector<typename numbers::NumberTraits<Number>::real_type> Sigma_r;

  /**
   * Perform RSVD for the matrix and return U and VT into A and B respectively.
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


#endif /* INCLUDE_LAPACK_FULL_MATRIX_EXT_H_ */
