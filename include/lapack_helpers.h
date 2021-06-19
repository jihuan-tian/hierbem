/**
 * \file lapack_helpers.h
 * \brief Exposes LAPACK helper functions defined in lapack_full_matrix.cc and
 * define new ones by following them as examples.
 * \ingroup linalg
 * \date 2021-06-09
 * \author Jihuan Tian
 */

#ifndef INCLUDE_LAPACK_HELPERS_H_
#define INCLUDE_LAPACK_HELPERS_H_

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

DEAL_II_NAMESPACE_OPEN

using namespace LAPACKSupport;

namespace internal
{
  namespace LAPACKFullMatrixImplementation
  {
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
                AlignedVector<T> &    matrix,
                const types::blas_int n_rows,
                std::vector<T> &      real_part_eigenvalues,
                std::vector<T> &      imag_part_eigenvalues,
                std::vector<T> &      left_eigenvectors,
                std::vector<T> &      right_eigenvectors,
                std::vector<T> &      real_work,
                std::vector<T> & /*complex_work*/,
                const types::blas_int work_flag,
                types::blas_int &     info);

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
                std::vector<T> &              real_work,
                const types::blas_int         work_flag,
                types::blas_int &             info);

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
                 AlignedVector<T> &    matrix,
                 std::vector<T> &      singular_values,
                 AlignedVector<T> &    left_vectors,
                 AlignedVector<T> &    right_vectors,
                 std::vector<T> &      real_work,
                 std::vector<T> & /*complex work*/,
                 std::vector<types::blas_int> &integer_work,
                 const types::blas_int         work_flag,
                 types::blas_int &             info);

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
                 std::vector<T> &                singular_values,
                 AlignedVector<std::complex<T>> &left_vectors,
                 AlignedVector<std::complex<T>> &right_vectors,
                 std::vector<std::complex<T>> &  work,
                 std::vector<T> &                real_work,
                 std::vector<types::blas_int> &  integer_work,
                 const types::blas_int &         work_flag,
                 types::blas_int &               info);
  } // namespace LAPACKFullMatrixImplementation
} // namespace internal

DEAL_II_NAMESPACE_CLOSE
#endif /* INCLUDE_LAPACK_HELPERS_H_ */
