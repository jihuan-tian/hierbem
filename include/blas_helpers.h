/**
 * @file blas_helpers.h
 * @brief Helper functions for calling BLAS functions.
 * @ingroup linalg
 *
 * @date 2025-03-28
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_BLAS_HELPERS_H_
#define HIERBEM_INCLUDE_BLAS_HELPERS_H_

#include <deal.II/base/aligned_vector.h>

#include <deal.II/lac/lapack_support.h>

#include <type_traits>

#include "blas_templates.h"
#include "config.h"

HBEM_NS_OPEN

namespace BLASHelpers
{
  template <typename T>
  void
  scal_helper(const dealii::types::blas_int n,
              const T                       alpha,
              dealii::AlignedVector<T>     &array,
              const dealii::types::blas_int incx = 1)
  {
    static_assert(
      std::is_same<T, double>::value || std::is_same<T, float>::value ||
        std::is_same<T, std::complex<double>>::value ||
        std::is_same<T, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");
    scal(&n, &alpha, array.data(), &incx);
  }
} // namespace BLASHelpers

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BLAS_HELPERS_H_
