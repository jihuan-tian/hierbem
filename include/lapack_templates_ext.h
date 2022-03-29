/**
 * \file lapack_templates_ext.h
 * \brief Append more template based LAPACK functions to those existing in
 * deal.ii.
 *
 * \mynote{In this file, LAPACK raw functions are declared as @p extern "C". C++
 * overloaded functions are defined based on the four data types, float, double,
 * complex<float> and complex<double>.}
 *
 * \date 2021-10-13
 * \author Jihuan Tian
 */
#ifndef INCLUDE_LAPACK_TEMPLATES_EXT_H_
#define INCLUDE_LAPACK_TEMPLATES_EXT_H_

#include <deal.II/lac/lapack_support.h>

#ifdef DEAL_II_HAVE_FP_EXCEPTIONS
#  include <cfenv>
#endif

extern "C"
{
  /**
   * Solve a triangular system.
   *
   * \ingroup lapack
   *
   * @param uplo
   * @param trans
   * @param diag
   * @param n
   * @param a
   * @param lda
   * @param x
   * @param incx
   */
  void
  strsv_(const char *                   uplo,
         const char *                   trans,
         const char *                   diag,
         const dealii::types::blas_int *n,
         const float *                  a,
         const dealii::types::blas_int *lda,
         float *                        x,
         const dealii::types::blas_int *incx);

  void
  dtrsv_(const char *                   uplo,
         const char *                   trans,
         const char *                   diag,
         const dealii::types::blas_int *n,
         const double *                 a,
         const dealii::types::blas_int *lda,
         double *                       x,
         const dealii::types::blas_int *incx);

  void
  ctrsv_(const char *                   uplo,
         const char *                   trans,
         const char *                   diag,
         const dealii::types::blas_int *n,
         const std::complex<float> *    a,
         const dealii::types::blas_int *lda,
         std::complex<float> *          x,
         const dealii::types::blas_int *incx);

  void
  ztrsv_(const char *                   uplo,
         const char *                   trans,
         const char *                   diag,
         const dealii::types::blas_int *n,
         const std::complex<double> *   a,
         const dealii::types::blas_int *lda,
         std::complex<double> *         x,
         const dealii::types::blas_int *incx);
}


template <typename number1, typename number2>
inline void
trsv(const char *                   uplo,
     const char *                   trans,
     const char *                   diag,
     const dealii::types::blas_int *n,
     const number1 *                a,
     const dealii::types::blas_int *lda,
     number2 *                      x,
     const dealii::types::blas_int *incx)
{
  Assert(false, ExcNotImplemented());
}


inline void
trsv(const char *                   uplo,
     const char *                   trans,
     const char *                   diag,
     const dealii::types::blas_int *n,
     const float *                  a,
     const dealii::types::blas_int *lda,
     float *                        x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  strsv_(uplo, trans, diag, n, a, lda, x, incx);
#else
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("strsv"));
#endif
}


inline void
trsv(const char *                   uplo,
     const char *                   trans,
     const char *                   diag,
     const dealii::types::blas_int *n,
     const double *                 a,
     const dealii::types::blas_int *lda,
     double *                       x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  dtrsv_(uplo, trans, diag, n, a, lda, x, incx);
#else
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("dtrsv"));
#endif
}


inline void
trsv(const char *                   uplo,
     const char *                   trans,
     const char *                   diag,
     const dealii::types::blas_int *n,
     const std::complex<float> *    a,
     const dealii::types::blas_int *lda,
     std::complex<float> *          x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  ctrsv_(uplo, trans, diag, n, a, lda, x, incx);
#else
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("ctrsv"));
#endif
}


inline void
trsv(const char *                   uplo,
     const char *                   trans,
     const char *                   diag,
     const dealii::types::blas_int *n,
     const std::complex<double> *   a,
     const dealii::types::blas_int *lda,
     std::complex<double> *         x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  ztrsv_(uplo, trans, diag, n, a, lda, x, incx);
#else
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("ztrsv"));
#endif
}

#endif /* INCLUDE_LAPACK_TEMPLATES_EXT_H_ */
