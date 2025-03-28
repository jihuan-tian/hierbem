/**
 * @file blas_templates.h
 * @brief Template based BLAS functions.
 * @ingroup linalg
 *
 * @date 2025-03-28
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_BLAS_TEMPLATES_H_
#define HIERBEM_INCLUDE_BLAS_TEMPLATES_H_

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/lapack_support.h>

#include <complex>

#ifdef DEAL_II_HAVE_FP_EXCEPTIONS
#  include <cfenv>
#endif

#include "config.h"

using namespace dealii;

extern "C"
{
  /**
   * Scale a vector by a factor.
   */
  void DEAL_II_FORTRAN_MANGLE(sscal,
                              SSCAL)(const dealii::types::blas_int *n,
                                     const float                   *alpha,
                                     float                         *x,
                                     const dealii::types::blas_int *incx);

  void DEAL_II_FORTRAN_MANGLE(dscal,
                              DSCAL)(const dealii::types::blas_int *n,
                                     const double                  *alpha,
                                     double                        *x,
                                     const dealii::types::blas_int *incx);

  void DEAL_II_FORTRAN_MANGLE(cscal,
                              CSCAL)(const dealii::types::blas_int *n,
                                     const std::complex<float>     *alpha,
                                     std::complex<float>           *x,
                                     const dealii::types::blas_int *incx);

  void DEAL_II_FORTRAN_MANGLE(zscal,
                              ZSCAL)(const dealii::types::blas_int *n,
                                     const std::complex<double>    *alpha,
                                     std::complex<double>          *x,
                                     const dealii::types::blas_int *incx);
}

HBEM_NS_OPEN

template <typename number1, typename number2>
inline void
scal(const dealii::types::blas_int *n,
     const number1                 *alpha,
     number2                       *x,
     const dealii::types::blas_int *incx)
{
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  Assert(false, ExcNotImplemented());
}


inline void
scal(const dealii::types::blas_int *n,
     const float                   *alpha,
     float                         *x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(sscal, SSCAL)(n, alpha, x, incx);
#else
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("sscal"));
#endif
}


inline void
scal(const dealii::types::blas_int *n,
     const double                  *alpha,
     double                        *x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(dscal, DSCAL)(n, alpha, x, incx);
#else
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("dscal"));
#endif
}


inline void
scal(const dealii::types::blas_int *n,
     const std::complex<float>     *alpha,
     std::complex<float>           *x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(cscal, CSCAL)(n, alpha, x, incx);
#else
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("cscal"));
#endif
}


inline void
scal(const dealii::types::blas_int *n,
     const std::complex<double>    *alpha,
     std::complex<double>          *x,
     const dealii::types::blas_int *incx)
{
#ifdef DEAL_II_WITH_LAPACK
  DEAL_II_FORTRAN_MANGLE(zscal, ZSCAL)(n, alpha, x, incx);
#else
  (void)n;
  (void)alpha;
  (void)x;
  (void)incx;
  Assert(false, LAPACKSupport::ExcMissing("zscal"));
#endif
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BLAS_TEMPLATES_H_
