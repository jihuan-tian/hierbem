/**
 * \file lapack_helpers.h
 * \brief Exposes LAPACK helper functions defined in @p lapack_full_matrix.cc and
 * define new ones by following them as examples. \mynote{Because helper
 * functions in @p lapack_full_matrix.cc are template functions and they are not
 * defined in a header file, their instantiation is limited within the
 * translation unit @p lapack_full_matrix.cc. Therefore, the helper functions
 * are not available in the deal.ii dynamic library compiled in Release mode.
 * But why they are available in the Debug mode library is still a mystery.
 * Comment on 2022-03-23 by Jihuan Tian.}
 *
 * \ingroup linalg
 * \date 2021-06-09
 * \author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_LAPACK_HELPERS_H_
#define HIERBEM_INCLUDE_LAPACK_HELPERS_H_

#include <deal.II/base/numbers.h>

#include <deal.II/lac/blas_extension_templates.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/lapack_templates.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/utilities.h>
#include <deal.II/lac/vector.h>

#include <complex>
#include <type_traits>

#include "config.h"
#include "lapack_templates_ext.h"

HBEM_NS_OPEN

using namespace dealii;
using namespace LAPACKSupport;

namespace LAPACKHelpers
{
  /**
   * @brief Multiplication of a general matrix and a vector.
   *
   * @tparam T
   * @param trans
   * @param alpha
   * @param n_rows
   * @param n_cols
   * @param matrix
   * @param x
   * @param beta
   * @param y
   * @param incx
   * @param incy
   */
  template <typename T>
  void
  gemv_helper(const char              trans,
              const T                 alpha,
              const types::blas_int   n_rows,
              const types::blas_int   n_cols,
              const AlignedVector<T> &matrix,
              const T                *x_pointer,
              const T                 beta,
              T                      *y_pointer,
              const types::blas_int   incx = 1,
              const types::blas_int   incy = 1)
  {
    static_assert(
      std::is_same<T, double>::value || std::is_same<T, float>::value ||
        std::is_same<T, std::complex<double>>::value ||
        std::is_same<T, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");
    Assert(trans == 'N' || trans == 'n' || trans == 'T' || trans == 't' ||
             trans == 'C' || trans == 'c',
           ExcInternalError());

    gemv(&trans,
         &n_rows,
         &n_cols,
         &alpha,
         matrix.data(),
         &n_rows,
         x_pointer,
         &incx,
         &beta,
         y_pointer,
         &incy);
  }

  /**
   * Multiplication of a symmetric matrix and a vector. Only the triangular part
   * of the symmetric matrix is stored.
   *
   * @param uplo
   * @param alpha
   * @param n_rows
   * @param matrix
   * @param x_pointer
   * @param beta
   * @param y_pointer
   * @param incx
   * @param incy
   */
  template <typename T>
  void
  symv_helper(const char              uplo,
              const T                 alpha,
              const types::blas_int   n_rows,
              const AlignedVector<T> &matrix,
              const T                *x_pointer,
              const T                 beta,
              T                      *y_pointer,
              const types::blas_int   incx = 1,
              const types::blas_int   incy = 1)
  {
    static_assert(
      std::is_same<T, double>::value || std::is_same<T, float>::value ||
        std::is_same<T, std::complex<double>>::value ||
        std::is_same<T, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");
    Assert(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l',
           ExcInternalError());
    // The matrix should be square.
    Assert(static_cast<size_t>(n_rows * n_rows) == matrix.size(),
           ExcInternalError());

    symv(&uplo,
         &n_rows,
         &alpha,
         matrix.data(),
         &n_rows,
         x_pointer,
         &incx,
         &beta,
         y_pointer,
         &incy);
  }


  /**
   * Multiplication of an Hermite symmetric matrix and a vector. Only the
   * triangular part of the symmetric matrix is stored.
   *
   * @param uplo
   * @param alpha
   * @param n_rows
   * @param matrix
   * @param x_pointer
   * @param beta
   * @param y_pointer
   * @param incx
   * @param incy
   */
  template <typename T>
  void
  hemv_helper(const char              uplo,
              const T                 alpha,
              const types::blas_int   n_rows,
              const AlignedVector<T> &matrix,
              const T                *x_pointer,
              const T                 beta,
              T                      *y_pointer,
              const types::blas_int   incx = 1,
              const types::blas_int   incy = 1)
  {
    static_assert(
      std::is_same<T, std::complex<double>>::value ||
        std::is_same<T, std::complex<float>>::value,
      "Only implemented for std::complex<double> and std::complex<float>");
    Assert(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l',
           ExcInternalError());
    // The matrix should be square.
    Assert(static_cast<size_t>(n_rows * n_rows) == matrix.size(),
           ExcInternalError());

    hemv(&uplo,
         &n_rows,
         &alpha,
         matrix.data(),
         &n_rows,
         x_pointer,
         &incx,
         &beta,
         y_pointer,
         &incy);
  }


  /**
   * Multiplication of a triangular matrix with a vector.
   *
   * @param uplo
   * @param trans
   * @param diag
   * @param n_rows
   * @param matrix
   * @param x_pointer
   * @param incx
   */
  template <typename T>
  void
  trmv_helper(const char              uplo,
              const char              trans,
              const char              diag,
              const types::blas_int   n_rows,
              const AlignedVector<T> &matrix,
              T                      *x_pointer,
              const types::blas_int   incx = 1)
  {
    static_assert(
      std::is_same<T, double>::value || std::is_same<T, float>::value ||
        std::is_same<T, std::complex<double>>::value ||
        std::is_same<T, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");
    Assert(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l',
           ExcInternalError());
    Assert(trans == 'N' || trans == 'n' || trans == 'T' || trans == 't' ||
             trans == 'C' || trans == 'c',
           ExcInternalError());
    Assert(diag == 'U' || diag == 'u' || diag == 'N' || diag == 'n',
           ExcInternalError());
    trmv(
      &uplo, &trans, &diag, &n_rows, matrix.data(), &n_rows, x_pointer, &incx);
  }


  // ZGEEV/CGEEV and DGEEV/SGEEV need different work arrays and different
  // output arrays for eigenvalues. This makes working with generic scalar
  // types a bit difficult. To get around this, geev_helper has the same
  // signature for real and complex arguments, but it ignores some
  // parameters when called with a real type and ignores different
  // parameters when called with a complex type.
  template <typename T>
  void
  geev_helper(const char            vl,
              const char            vr,
              AlignedVector<T>     &matrix,
              const types::blas_int n_rows,
              std::vector<T>       &real_part_eigenvalues,
              std::vector<T>       &imag_part_eigenvalues,
              std::vector<T>       &left_eigenvectors,
              std::vector<T>       &right_eigenvectors,
              std::vector<T>       &real_work,
              std::vector<T> & /*complex_work*/,
              const types::blas_int work_flag,
              types::blas_int      &info)
  {
    static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, float>::value,
                  "Only implemented for double and float");
    Assert(matrix.size() == static_cast<std::size_t>(n_rows * n_rows),
           ExcInternalError());
    Assert(static_cast<std::size_t>(n_rows) <= real_part_eigenvalues.size(),
           ExcInternalError());
    Assert(static_cast<std::size_t>(n_rows) <= imag_part_eigenvalues.size(),
           ExcInternalError());
    if (vl == 'V')
      Assert(static_cast<std::size_t>(n_rows * n_rows) <=
               left_eigenvectors.size(),
             ExcInternalError());
    if (vr == 'V')
      Assert(static_cast<std::size_t>(n_rows * n_rows) <=
               right_eigenvectors.size(),
             ExcInternalError());
    Assert(work_flag == -1 ||
             static_cast<std::size_t>(2 * n_rows) <= real_work.size(),
           ExcInternalError());
    Assert(work_flag == -1 || std::max<long int>(1, 3 * n_rows) <= work_flag,
           ExcInternalError());
    geev(&vl,
         &vr,
         &n_rows,
         matrix.data(),
         &n_rows,
         real_part_eigenvalues.data(),
         imag_part_eigenvalues.data(),
         left_eigenvectors.data(),
         &n_rows,
         right_eigenvectors.data(),
         &n_rows,
         real_work.data(),
         &work_flag,
         &info);
  }

  template <typename T>
  void
  geev_helper(const char                      vl,
              const char                      vr,
              AlignedVector<std::complex<T>> &matrix,
              const types::blas_int           n_rows,
              std::vector<T> & /*real_part_eigenvalues*/,
              std::vector<std::complex<T>> &eigenvalues,
              std::vector<std::complex<T>> &left_eigenvectors,
              std::vector<std::complex<T>> &right_eigenvectors,
              std::vector<std::complex<T>> &complex_work,
              std::vector<T>               &real_work,
              const types::blas_int         work_flag,
              types::blas_int              &info)
  {
    static_assert(
      std::is_same<T, double>::value || std::is_same<T, float>::value,
      "Only implemented for std::complex<double> and std::complex<float>");
    Assert(matrix.size() == static_cast<std::size_t>(n_rows * n_rows),
           ExcInternalError());
    Assert(static_cast<std::size_t>(n_rows) <= eigenvalues.size(),
           ExcInternalError());
    if (vl == 'V')
      Assert(static_cast<std::size_t>(n_rows * n_rows) <=
               left_eigenvectors.size(),
             ExcInternalError());
    if (vr == 'V')
      Assert(static_cast<std::size_t>(n_rows * n_rows) <=
               right_eigenvectors.size(),
             ExcInternalError());
    Assert(std::max<std::size_t>(1, work_flag) <= real_work.size(),
           ExcInternalError());
    Assert(work_flag == -1 || std::max<long int>(1, 2 * n_rows) <= (work_flag),
           ExcInternalError());

    geev(&vl,
         &vr,
         &n_rows,
         matrix.data(),
         &n_rows,
         eigenvalues.data(),
         left_eigenvectors.data(),
         &n_rows,
         right_eigenvectors.data(),
         &n_rows,
         complex_work.data(),
         &work_flag,
         real_work.data(),
         &info);
  }

  /**
   * Helper function for real valued SVD.
   *
   * Its implementation has already existed in the deal.ii library.
   * @param job
   * @param n_rows
   * @param n_cols
   * @param matrix
   * @param singular_values
   * @param left_vectors
   * @param right_vectors
   * @param real_work
   * @param
   * @param integer_work
   * @param work_flag
   * @param info
   */
  template <typename T>
  void
  gesdd_helper(const char            job,
               const types::blas_int n_rows,
               const types::blas_int n_cols,
               AlignedVector<T>     &matrix,
               std::vector<T>       &singular_values,
               AlignedVector<T>     &left_vectors,
               AlignedVector<T>     &right_vectors,
               std::vector<T>       &real_work,
               std::vector<T> & /*complex work*/,
               std::vector<types::blas_int> &integer_work,
               const types::blas_int         work_flag,
               types::blas_int              &info)
  {
    static_assert(std::is_same<T, double>::value ||
                    std::is_same<T, float>::value,
                  "Only implemented for double and float");
    Assert(job == 'A' || job == 'S' || job == 'O' || job == 'N',
           ExcInternalError());
    Assert(static_cast<std::size_t>(n_rows * n_cols) == matrix.size(),
           ExcInternalError());
    Assert(std::min<std::size_t>(n_rows, n_cols) <= singular_values.size(),
           ExcInternalError());
    Assert(8 * std::min<std::size_t>(n_rows, n_cols) <= integer_work.size(),
           ExcInternalError());
    Assert(work_flag == -1 ||
             static_cast<std::size_t>(work_flag) <= real_work.size(),
           ExcInternalError());

    gesdd(&job,
          &n_rows,
          &n_cols,
          matrix.data(),
          &n_rows,
          singular_values.data(),
          left_vectors.data(),
          &n_rows,
          right_vectors.data(),
          &n_cols,
          real_work.data(),
          &work_flag,
          integer_work.data(),
          &info);
  }

  /**
   * Helper function for complex valued SVD.
   *
   * Its implementation has already existed in the deal.ii library.
   * @param job
   * @param n_rows
   * @param n_cols
   * @param matrix
   * @param singular_values
   * @param left_vectors
   * @param right_vectors
   * @param work
   * @param real_work
   * @param integer_work
   * @param work_flag
   * @param info
   */
  template <typename T>
  void
  gesdd_helper(const char                      job,
               const types::blas_int           n_rows,
               const types::blas_int           n_cols,
               AlignedVector<std::complex<T>> &matrix,
               std::vector<T>                 &singular_values,
               AlignedVector<std::complex<T>> &left_vectors,
               AlignedVector<std::complex<T>> &right_vectors,
               std::vector<std::complex<T>>   &work,
               std::vector<T>                 &real_work,
               std::vector<types::blas_int>   &integer_work,
               const types::blas_int          &work_flag,
               types::blas_int                &info)
  {
    static_assert(
      std::is_same<T, double>::value || std::is_same<T, float>::value,
      "Only implemented for std::complex<double> and std::complex<float>");
    Assert(job == 'A' || job == 'S' || job == 'O' || job == 'N',
           ExcInternalError());
    Assert(static_cast<std::size_t>(n_rows * n_cols) == matrix.size(),
           ExcInternalError());
    Assert(static_cast<std::size_t>(std::min(n_rows, n_cols)) <=
             singular_values.size(),
           ExcInternalError());
    Assert(8 * std::min<std::size_t>(n_rows, n_cols) <= integer_work.size(),
           ExcInternalError());
    Assert(work_flag == -1 ||
             static_cast<std::size_t>(work_flag) <= real_work.size(),
           ExcInternalError());

    gesdd(&job,
          &n_rows,
          &n_cols,
          matrix.data(),
          &n_rows,
          singular_values.data(),
          left_vectors.data(),
          &n_rows,
          right_vectors.data(),
          &n_cols,
          work.data(),
          &work_flag,
          real_work.data(),
          integer_work.data(),
          &info);
  }

  /**
   * Helper function for QR decomposition.
   */
  template <typename T>
  void
  geqrf_helper(const types::blas_int n_rows,
               const types::blas_int n_cols,
               AlignedVector<T>     &matrix,
               std::vector<T>       &tau,
               std::vector<T>       &work,
               const types::blas_int work_flag,
               types::blas_int      &info)
  {
    Assert(static_cast<std::size_t>(n_rows * n_cols) == matrix.size(),
           ExcInternalError());
    Assert(tau.size() == static_cast<size_t>(std::min(n_rows, n_cols)),
           ExcInternalError());
    Assert(work_flag == -1 || static_cast<size_t>(work_flag) >=
                                static_cast<size_t>(std::max(1, n_cols)),
           ExcInternalError());

    geqrf(&n_rows,
          &n_cols,
          matrix.data(),
          &n_rows,
          tau.data(),
          work.data(),
          &work_flag,
          &info);
  }


  /**
   * Real valued version of the helper function for \p trsv, which solves a
   * unit or non-unit, upper or lower triangular matrix.
   *
   * \mynote{1. Access the matrix data: the internal data contained within a
   * @p LAPACKFullMatrixExt has the type @p AlignedVector, which is a
   * @p protected member variable in the parent class @p TableBase of
   * @p LAPACKFullMatrixExt. Therefore, this variable can be directly visited
   * from @p LAPACKFullMatrixExt and passed into this function.
   * 2. Access the RHS vector data: being different from
   * @p LAPACKFullMatrixExt, the @p values member variable of type
   * @p AlignedVector contained in the RHS vector is @p private, which cannot
   * be accessed from @p LAPACKFullMatrixExt. Therefore, the pointer to the
   * internal data within the RHS vector can be obtained by calling the member
   * function @p Vector::data(). Then it is passed to this function.}
   *
   * @param uplo
   * @param is_transposed
   * @param is_hermite_transposed
   * @param is_unit_diagonal
   * @param n_rows
   * @param matrix
   * @param right_vector
   * @param incx
   */
  template <typename T>
  void
  trsv_helper(const char              uplo,
              const bool              is_transposed,
              const bool              is_hermite_transposed,
              const bool              is_unit_diagonal,
              const types::blas_int   n_rows,
              const AlignedVector<T> &matrix,
              T                      *right_vector_pointer,
              const types::blas_int   incx = 1)
  {
    // The matrix should be square.
    Assert(static_cast<size_t>(n_rows * n_rows) == matrix.size(),
           ExcInternalError());

    (void)is_hermite_transposed;

    const char trans(is_transposed ? 'T' : 'N');
    const char diag(is_unit_diagonal ? 'U' : 'N');

    trsv(&uplo,
         &trans,
         &diag,
         &n_rows,
         matrix.data(),
         &n_rows,
         right_vector_pointer,
         &incx);
  }


  /**
   * Complex valued version of the helper function for \p trsv.
   *
   * \alert{Since the \p values member variable of type \p AlignedVector
   * stored in the right hand side vector is private, which cannot be directly
   * accessed from \p LAPACKFullMatrixExt, the data pointer to the right hand
   * side vector is passed as argument, which can be obtained via the member
   * function \p data of \p Vector.}
   *
   * @param uplo
   * @param is_transposed
   * @param is_hermite_transposed
   * @param is_unit_diagonal
   * @param n_rows
   * @param matrix
   * @param right_vector
   * @param incx
   */
  template <typename T>
  void
  trsv_helper(const char                            uplo,
              const bool                            is_transposed,
              const bool                            is_hermite_transposed,
              const bool                            is_unit_diagonal,
              const types::blas_int                 n_rows,
              const AlignedVector<std::complex<T>> &matrix,
              std::complex<T>                      *right_vector_pointer,
              const types::blas_int                 incx = 1)
  {
    Assert(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l',
           ExcInternalError());
    // The matrix should be square.
    Assert(static_cast<size_t>(n_rows * n_rows) == matrix.size(),
           ExcInternalError());

    // N.B. When the number field \f$\mathbb{K}\f$ for matrix is complex, the
    // transposition of a matrix is Hermitian, i.e. including both the normal
    // transposition and complex conjugation.
    char trans;
    if (is_transposed)
      trans = is_hermite_transposed ? 'C' : 'T';
    else
      trans = 'N';

    const char diag(is_unit_diagonal ? 'U' : 'N');

    trsv(&uplo,
         &trans,
         &diag,
         &n_rows,
         matrix.data(), // Get the data pointer to the \p AlignedVector.
         &n_rows,
         right_vector_pointer,
         &incx);
  }
} // namespace LAPACKHelpers

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LAPACK_HELPERS_H_
