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
#ifndef HIERBEM_INCLUDE_LAPACK_TEMPLATES_EXT_H_
#define HIERBEM_INCLUDE_LAPACK_TEMPLATES_EXT_H_

#include <deal.II/base/config.h>

#include <deal.II/lac/lapack_support.h>

#include <complex>

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
  void DEAL_II_FORTRAN_MANGLE(ssymv,
                              SSYMV)(const char                    *uplo,
                                     const dealii::types::blas_int *n,
                                     const float                   *alpha,
                                     const float                   *a,
                                     const dealii::types::blas_int *lda,
                                     const float                   *x,
                                     const dealii::types::blas_int *incx,
                                     const float                   *beta,
                                     float                         *y,
                                     const dealii::types::blas_int *incy);

  void DEAL_II_FORTRAN_MANGLE(dsymv,
                              DSYMV)(const char                    *uplo,
                                     const dealii::types::blas_int *n,
                                     const double                  *alpha,
                                     const double                  *a,
                                     const dealii::types::blas_int *lda,
                                     const double                  *x,
                                     const dealii::types::blas_int *incx,
                                     const double                  *beta,
                                     double                        *y,
                                     const dealii::types::blas_int *incy);

  void DEAL_II_FORTRAN_MANGLE(csymv,
                              CSYMV)(const char                    *uplo,
                                     const dealii::types::blas_int *n,
                                     const std::complex<float>     *alpha,
                                     const std::complex<float>     *a,
                                     const dealii::types::blas_int *lda,
                                     const std::complex<float>     *x,
                                     const dealii::types::blas_int *incx,
                                     const std::complex<float>     *beta,
                                     std::complex<float>           *y,
                                     const dealii::types::blas_int *incy);

  void DEAL_II_FORTRAN_MANGLE(zsymv,
                              ZSYMV)(const char                    *uplo,
                                     const dealii::types::blas_int *n,
                                     const std::complex<double>    *alpha,
                                     const std::complex<double>    *a,
                                     const dealii::types::blas_int *lda,
                                     const std::complex<double>    *x,
                                     const dealii::types::blas_int *incx,
                                     const std::complex<double>    *beta,
                                     std::complex<double>          *y,
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
  void DEAL_II_FORTRAN_MANGLE(strsv,
                              STRSV)(const char                    *uplo,
                                     const char                    *trans,
                                     const char                    *diag,
                                     const dealii::types::blas_int *n,
                                     const float                   *a,
                                     const dealii::types::blas_int *lda,
                                     float                         *x,
                                     const dealii::types::blas_int *incx);

  void DEAL_II_FORTRAN_MANGLE(dtrsv,
                              DTRSV)(const char                    *uplo,
                                     const char                    *trans,
                                     const char                    *diag,
                                     const dealii::types::blas_int *n,
                                     const double                  *a,
                                     const dealii::types::blas_int *lda,
                                     double                        *x,
                                     const dealii::types::blas_int *incx);

  void DEAL_II_FORTRAN_MANGLE(ctrsv,
                              CTRSV)(const char                    *uplo,
                                     const char                    *trans,
                                     const char                    *diag,
                                     const dealii::types::blas_int *n,
                                     const std::complex<float>     *a,
                                     const dealii::types::blas_int *lda,
                                     std::complex<float>           *x,
                                     const dealii::types::blas_int *incx);

  void DEAL_II_FORTRAN_MANGLE(ztrsv,
                              ZTRSV)(const char                    *uplo,
                                     const char                    *trans,
                                     const char                    *diag,
                                     const dealii::types::blas_int *n,
                                     const std::complex<double>    *a,
                                     const dealii::types::blas_int *lda,
                                     std::complex<double>          *x,
                                     const dealii::types::blas_int *incx);

  /**
   * Perform Hermitian matrix multiplication: <code>C := alpha*A*A**H +
   * beta*C</code> or <code>C := alpha*A**H*A + beta*C</code>. Only complexed
   * value versions are available.
   */
  void DEAL_II_FORTRAN_MANGLE(cherk, CHERK)(const char *uplo,
                                            const char *trans,
                                            const dealii::types::blas_int *n,
                                            const dealii::types::blas_int *k,
                                            const float               *alpha,
                                            const std::complex<float> *a,
                                            const dealii::types::blas_int *lda,
                                            const float                   *beta,
                                            std::complex<float>           *c,
                                            const dealii::types::blas_int *ldc);

  void DEAL_II_FORTRAN_MANGLE(zherk, ZHERK)(const char *uplo,
                                            const char *trans,
                                            const dealii::types::blas_int *n,
                                            const dealii::types::blas_int *k,
                                            const double               *alpha,
                                            const std::complex<double> *a,
                                            const dealii::types::blas_int *lda,
                                            const double                  *beta,
                                            std::complex<double>          *c,
                                            const dealii::types::blas_int *ldc);
}


template <typename number1,
          typename number2,
          typename number3,
          typename number4,
          typename number5>
inline void
symv(const char                    *uplo,
     const dealii::types::blas_int *n,
     const number1                 *alpha,
     const number2                 *a,
     const dealii::types::blas_int *lda,
     const number3                 *x,
     const dealii::types::blas_int *incx,
     const number4                 *beta,
     number5                       *y,
     const dealii::types::blas_int *incy)
{
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
  Assert(false, ExcNotImplemented());
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
  DEAL_II_FORTRAN_MANGLE(ssymv, SSYMV)
  (uplo, n, alpha, a, lda, x, incx, beta, y, incy);
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
  DEAL_II_FORTRAN_MANGLE(dsymv, DSYMV)
  (uplo, n, alpha, a, lda, x, incx, beta, y, incy);
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


inline void
symv(const char                    *uplo,
     const dealii::types::blas_int *n,
     const std::complex<float>     *alpha,
     const std::complex<float>     *a,
     const dealii::types::blas_int *lda,
     const std::complex<float>     *x,
     const dealii::types::blas_int *incx,
     const std::complex<float>     *beta,
     std::complex<float>           *y,
     const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(csymv, CSYMV)
  (uplo, n, alpha, a, lda, x, incx, beta, y, incy);
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
  Assert(false, LAPACKSupport::ExcMissing("csymv"));
#endif
}


inline void
symv(const char                    *uplo,
     const dealii::types::blas_int *n,
     const std::complex<double>    *alpha,
     const std::complex<double>    *a,
     const dealii::types::blas_int *lda,
     const std::complex<double>    *x,
     const dealii::types::blas_int *incx,
     const std::complex<double>    *beta,
     std::complex<double>          *y,
     const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(zsymv, ZSYMV)
  (uplo, n, alpha, a, lda, x, incx, beta, y, incy);
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
  Assert(false, LAPACKSupport::ExcMissing("zsymv"));
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
  (void)uplo;
  (void)trans;
  (void)diag;
  (void)n;
  (void)a;
  (void)lda;
  (void)x;
  (void)incx;
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
  DEAL_II_FORTRAN_MANGLE(strsv, STRSV)(uplo, trans, diag, n, a, lda, x, incx);
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
  DEAL_II_FORTRAN_MANGLE(dtrsv, DTRSV)(uplo, trans, diag, n, a, lda, x, incx);
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
  DEAL_II_FORTRAN_MANGLE(ctrsv, CTRSV)(uplo, trans, diag, n, a, lda, x, incx);
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
  DEAL_II_FORTRAN_MANGLE(ztrsv, ZTRSV)(uplo, trans, diag, n, a, lda, x, incx);
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


inline void
herk(const char                    *uplo,
     const char                    *trans,
     const dealii::types::blas_int *n,
     const dealii::types::blas_int *k,
     const float                   *alpha,
     const std::complex<float>     *a,
     const dealii::types::blas_int *lda,
     const float                   *beta,
     std::complex<float>           *c,
     const dealii::types::blas_int *ldc)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(cherk, CHERK)
  (uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
#else
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)beta;
  (void)c;
  (void)ldc;
  Assert(false, LAPACKSupport::ExcMissing("cherk"));
#endif
}


inline void
herk(const char                    *uplo,
     const char                    *trans,
     const dealii::types::blas_int *n,
     const dealii::types::blas_int *k,
     const double                  *alpha,
     const std::complex<double>    *a,
     const dealii::types::blas_int *lda,
     const double                  *beta,
     std::complex<double>          *c,
     const dealii::types::blas_int *ldc)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(zherk, ZHERK)
  (uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
#else
  (void)uplo;
  (void)trans;
  (void)n;
  (void)k;
  (void)alpha;
  (void)a;
  (void)lda;
  (void)beta;
  (void)c;
  (void)ldc;
  Assert(false, LAPACKSupport::ExcMissing("zherk"));
#endif
}

#endif // HIERBEM_INCLUDE_LAPACK_TEMPLATES_EXT_H_
