// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file blas_templates.h
 * @brief Template based BLAS functions.
 * @ingroup linalg
 *
 * @date 2025-03-28
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_LINEAR_ALGEBRA_BLAS_TEMPLATES_H_
#define HIERBEM_INCLUDE_LINEAR_ALGEBRA_BLAS_TEMPLATES_H_

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
   * Compute the dot product of two vectors.
   */
  float DEAL_II_FORTRAN_MANGLE(sdot, SDOT)(const dealii::types::blas_int *n,
                                           const float                   *x,
                                           const dealii::types::blas_int *incx,
                                           const float                   *y,
                                           const dealii::types::blas_int *incy);

  double DEAL_II_FORTRAN_MANGLE(ddot,
                                DDOT)(const dealii::types::blas_int *n,
                                      const double                  *x,
                                      const dealii::types::blas_int *incx,
                                      const double                  *y,
                                      const dealii::types::blas_int *incy);

  /**
   * N.B. Complex conjugation is applied to x.
   */
  std::complex<float>
    DEAL_II_FORTRAN_MANGLE(cdotc, CDOTC)(const dealii::types::blas_int *n,
                                         const std::complex<float>     *x,
                                         const dealii::types::blas_int *incx,
                                         const std::complex<float>     *y,
                                         const dealii::types::blas_int *incy);

  /**
   * N.B. No complex conjugation is applied.
   */
  std::complex<float>
    DEAL_II_FORTRAN_MANGLE(cdotu, CDOTU)(const dealii::types::blas_int *n,
                                         const std::complex<float>     *x,
                                         const dealii::types::blas_int *incx,
                                         const std::complex<float>     *y,
                                         const dealii::types::blas_int *incy);

  /**
   * N.B. Complex conjugation is applied to x.
   */
  std::complex<double>
    DEAL_II_FORTRAN_MANGLE(zdotc, ZDOTC)(const dealii::types::blas_int *n,
                                         const std::complex<double>    *x,
                                         const dealii::types::blas_int *incx,
                                         const std::complex<double>    *y,
                                         const dealii::types::blas_int *incy);

  /**
   * N.B. No complex conjugation is applied.
   */
  std::complex<double>
    DEAL_II_FORTRAN_MANGLE(zdotu, ZDOTU)(const dealii::types::blas_int *n,
                                         const std::complex<double>    *x,
                                         const dealii::types::blas_int *incx,
                                         const std::complex<double>    *y,
                                         const dealii::types::blas_int *incy);

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

inline float
inner_product(const dealii::types::blas_int *n,
              const float                   *x,
              const dealii::types::blas_int *incx,
              const float                   *y,
              const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(sdot, SDOT)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("sdot"));
  return 0.;
#endif
}


inline double
inner_product(const dealii::types::blas_int *n,
              const double                  *x,
              const dealii::types::blas_int *incx,
              const double                  *y,
              const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(ddot, DDOT)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("ddot"));
  return 0.;
#endif
}


inline std::complex<float>
inner_product(const dealii::types::blas_int *n,
              const std::complex<float>     *x,
              const dealii::types::blas_int *incx,
              const std::complex<float>     *y,
              const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(cdotc, CDOTC)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("cdotc"));
  return std::complex<float>(0.);
#endif
}


inline std::complex<double>
inner_product(const dealii::types::blas_int *n,
              const std::complex<double>    *x,
              const dealii::types::blas_int *incx,
              const std::complex<double>    *y,
              const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(zdotc, ZDOTC)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("zdotc"));
  return std::complex<double>(0.);
#endif
}


inline float
linear_combination(const dealii::types::blas_int *n,
                   const float                   *x,
                   const dealii::types::blas_int *incx,
                   const float                   *y,
                   const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(sdot, SDOT)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("sdot"));
  return 0.;
#endif
}


inline double
linear_combination(const dealii::types::blas_int *n,
                   const double                  *x,
                   const dealii::types::blas_int *incx,
                   const double                  *y,
                   const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(ddot, DDOT)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("ddot"));
  return 0.;
#endif
}


inline std::complex<float>
linear_combination(const dealii::types::blas_int *n,
                   const std::complex<float>     *x,
                   const dealii::types::blas_int *incx,
                   const std::complex<float>     *y,
                   const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(cdotu, CDOTU)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("cdotu"));
  return std::complex<float>(0.);
#endif
}


inline std::complex<double>
linear_combination(const dealii::types::blas_int *n,
                   const std::complex<double>    *x,
                   const dealii::types::blas_int *incx,
                   const std::complex<double>    *y,
                   const dealii::types::blas_int *incy)
{
#ifdef DEAL_II_WITH_LAPACK
  return DEAL_II_FORTRAN_MANGLE(zdotu, ZDOTU)(n, x, incx, y, incy);
#else
  (void)n;
  (void)x;
  (void)incx;
  (void)y;
  (void)incy;

  Assert(false, LAPACKSupport::ExcMissing("zdotu"));
  return std::complex<double>(0.);
#endif
}


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

#endif // HIERBEM_INCLUDE_LINEAR_ALGEBRA_BLAS_TEMPLATES_H_
