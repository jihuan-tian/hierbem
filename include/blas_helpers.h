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
#include <deal.II/base/numbers.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/vector_operations_internal.h>

#include <type_traits>

#include "blas_templates.h"
#include "config.h"

HBEM_NS_OPEN

namespace BLASHelpers
{
  using size_type = dealii::types::blas_int;

  /**
   * @brief Compute the inner product of x and y. When in the complex valued
   * case, complex conjugation is applied to y.
   *
   * @tparam VectorType
   * @param n
   * @param x
   * @param y
   * @param incx
   * @param incy
   * @return
   */
  template <typename VectorType>
  typename VectorType::value_type
  inner_product_helper(const size_type   n,
                       const VectorType &x,
                       const VectorType &y,
                       const size_type   incx = 1,
                       const size_type   incy = 1)
  {
    using Number = typename VectorType::value_type;

    static_assert(
      std::is_same<Number, double>::value ||
        std::is_same<Number, float>::value ||
        std::is_same<Number, std::complex<double>>::value ||
        std::is_same<Number, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");

    // In BLAS, complex conjugation is applied to the first operand, therefore,
    // we put y before x.
    return inner_product(&n, y.data(), &incy, x.data(), &incx);
  }


  /**
   * @brief Compute the linear combination of x and y. When in the complex
   * valued case, there is no complex conjugation applied to y.
   *
   * @tparam VectorType
   * @param n
   * @param x
   * @param y
   * @param incx
   * @param incy
   * @return
   */
  template <typename VectorType>
  typename VectorType::value_type
  linear_combination_helper(const size_type   n,
                            const VectorType &x,
                            const VectorType &y,
                            const size_type   incx = 1,
                            const size_type   incy = 1)
  {
    using Number = typename VectorType::value_type;

    static_assert(
      std::is_same<Number, double>::value ||
        std::is_same<Number, float>::value ||
        std::is_same<Number, std::complex<double>>::value ||
        std::is_same<Number, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");

    return linear_combination(&n, x.data(), &incy, y.data(), &incx);
  }


  /**
   * @brief Scale a vector.
   *
   * @tparam Number
   * @param n
   * @param alpha
   * @param array
   * @param incx
   */
  template <typename Number, template <typename> typename VectorType>
  void
  scal_helper(const size_type     n,
              const Number        alpha,
              VectorType<Number> &array,
              const size_type     incx = 1)
  {
    static_assert(
      std::is_same<Number, double>::value ||
        std::is_same<Number, float>::value ||
        std::is_same<Number, std::complex<double>>::value ||
        std::is_same<Number, std::complex<float>>::value,
      "Only implemented for double, float, std::complex<double> and std::complex<float>");

    scal(&n, &alpha, array.data(), &incx);
  }
} // namespace BLASHelpers

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BLAS_HELPERS_H_
