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

using namespace dealii;

extern "C"
{
  /**
   * Perform matrix-vector multiplication \f$y:=\alpha A x + \beta y\f$, where
   * the matrix \f$A\f$ is symmetric.
   *
   * @param uplo
   * @param n
   * @param alpha
   * @param a
   * @param dla
   * @param x
   * @param incx
   * @param beta
   * @param y
   * @param incx
   */
  void
  ssymv_(const char                    *uplo,
         const dealii::types::blas_int *n,
         const float                   *alpha,
         const float                   *a,
         const dealii::types::blas_int *lda,
         const float                   *x,
         const dealii::types::blas_int *incx,
         const float                   *beta,
         float                         *y,
         const dealii::types::blas_int *incy);

  void
  dsymv_(const char                    *uplo,
         const dealii::types::blas_int *n,
         const double                  *alpha,
         const double                  *a,
         const dealii::types::blas_int *lda,
         const double                  *x,
         const dealii::types::blas_int *incx,
         const double                  *beta,
         double                        *y,
         const dealii::types::blas_int *incy);

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
  strsv_(const char                    *uplo,
         const char                    *trans,
         const char                    *diag,
         const dealii::types::blas_int *n,
         const float                   *a,
         const dealii::types::blas_int *lda,
         float                         *x,
         const dealii::types::blas_int *incx);

  void
  dtrsv_(const char                    *uplo,
         const char                    *trans,
         const char                    *diag,
         const dealii::types::blas_int *n,
         const double                  *a,
         const dealii::types::blas_int *lda,
         double                        *x,
         const dealii::types::blas_int *incx);

  void
  ctrsv_(const char                    *uplo,
         const char                    *trans,
         const char                    *diag,
         const dealii::types::blas_int *n,
         const std::complex<float>     *a,
         const dealii::types::blas_int *lda,
         std::complex<float>           *x,
         const dealii::types::blas_int *incx);

  void
  ztrsv_(const char                    *uplo,
         const char                    *trans,
         const char                    *diag,
         const dealii::types::blas_int *n,
         const std::complex<double>    *a,
         const dealii::types::blas_int *lda,
         std::complex<double>          *x,
         const dealii::types::blas_int *incx);
}


inline void
symv(const char                    *uplo,
     const dealii::types::blas_int *n,
     const float                   *alpha,
     const float                   *a,
     const dealii::types::blas_int *lda,
     const float                   *x,
     const dealii::types::blas_int *incx,
     const float                   *beta,
     float                         *y,
     const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  ssymv_(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
#else
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  Assert(false, LAPACKSupport::ExcMissing("ssymv"));
#endif
}


inline void
symv(const char                    *uplo,
     const dealii::types::blas_int *n,
     const double                  *alpha,
     const double                  *a,
     const dealii::types::blas_int *lda,
     const double                  *x,
     const dealii::types::blas_int *incx,
     const double                  *beta,
     double                        *y,
     const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  dsymv_(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
#else
  (void)uplo;
  (void)n;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
  (void)beta;
  (void)y;
  (void)incy;
  Assert(false, LAPACKSupport::ExcMissing("dsymv"));
#endif
}


template <typename number1, typename number2>
inline void
trsv(const char                    *uplo,
     const char                    *trans,
     const char                    *diag,
     const dealii::types::blas_int *n,
     const number1                 *a,
     const dealii::types::blas_int *lda,
     number2                       *x,
     const dealii::types::blas_int *incx)
{
  Assert(false, ExcNotImplemented());
}


inline void
trsv(const char                    *uplo,
     const char                    *trans,
     const char                    *diag,
     const dealii::types::blas_int *n,
     const float                   *a,
     const dealii::types::blas_int *lda,
     float                         *x,
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
trsv(const char                    *uplo,
     const char                    *trans,
     const char                    *diag,
     const dealii::types::blas_int *n,
     const double                  *a,
     const dealii::types::blas_int *lda,
     double                        *x,
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
trsv(const char                    *uplo,
     const char                    *trans,
     const char                    *diag,
     const dealii::types::blas_int *n,
     const std::complex<float>     *a,
     const dealii::types::blas_int *lda,
     std::complex<float>           *x,
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
trsv(const char                    *uplo,
     const char                    *trans,
     const char                    *diag,
     const dealii::types::blas_int *n,
     const std::complex<double>    *a,
     const dealii::types::blas_int *lda,
     std::complex<double>          *x,
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
