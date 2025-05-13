/*
 * linalg.h
 *
 *  Created on: 2020年11月30日
 *      Author: jihuan
 */

#ifndef HIERBEM_INCLUDE_LINALG_H_
#define HIERBEM_INCLUDE_LINALG_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/types.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operations_internal.h>

#include <tbb/partitioner.h>

#include <complex>
#include <limits>
#include <type_traits>

#include "blas_helpers.h"
#include "config.h"
#include "lapack_full_matrix_ext.h"
#include "number_traits.h"

HBEM_NS_OPEN

namespace LinAlg
{
  using namespace dealii;
  using size_type = types::blas_int;

  /**
   * Calculate the determinant of a $4 \times 4$ matrix.
   */
  template <typename number>
  number
  determinant4x4(const FullMatrix<number> &matrix)
  {
    Assert(matrix.m() == 4, ExcDimensionMismatch(matrix.m(), 4));
    Assert(matrix.n() == 4, ExcDimensionMismatch(matrix.n(), 4));

    return matrix(0, 0) *
             (matrix(1, 1) *
                (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) -
              matrix(1, 2) *
                (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) +
              matrix(1, 3) *
                (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1))) -
           matrix(0, 1) *
             (matrix(1, 0) *
                (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) -
              matrix(1, 2) *
                (matrix(2, 0) * matrix(3, 3) - matrix(2, 3) * matrix(3, 0)) +
              matrix(1, 3) *
                (matrix(2, 0) * matrix(3, 2) - matrix(2, 2) * matrix(3, 0))) +
           matrix(0, 2) *
             (matrix(1, 0) *
                (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) -
              matrix(1, 1) *
                (matrix(2, 0) * matrix(3, 3) - matrix(2, 3) * matrix(3, 0)) +
              matrix(1, 3) *
                (matrix(2, 0) * matrix(3, 1) - matrix(2, 1) * matrix(3, 0))) -
           matrix(0, 3) *
             (matrix(1, 0) *
                (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1)) -
              matrix(1, 1) *
                (matrix(2, 0) * matrix(3, 2) - matrix(2, 2) * matrix(3, 0)) +
              matrix(1, 2) *
                (matrix(2, 0) * matrix(3, 1) - matrix(2, 1) * matrix(3, 0)));
  }


  template <typename number>
  number
  determinant4x4(const LAPACKFullMatrixExt<number> &matrix)
  {
    Assert(matrix.m() == 4, ExcDimensionMismatch(matrix.m(), 4));
    Assert(matrix.n() == 4, ExcDimensionMismatch(matrix.n(), 4));

    return matrix(0, 0) *
             (matrix(1, 1) *
                (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) -
              matrix(1, 2) *
                (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) +
              matrix(1, 3) *
                (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1))) -
           matrix(0, 1) *
             (matrix(1, 0) *
                (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) -
              matrix(1, 2) *
                (matrix(2, 0) * matrix(3, 3) - matrix(2, 3) * matrix(3, 0)) +
              matrix(1, 3) *
                (matrix(2, 0) * matrix(3, 2) - matrix(2, 2) * matrix(3, 0))) +
           matrix(0, 2) *
             (matrix(1, 0) *
                (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) -
              matrix(1, 1) *
                (matrix(2, 0) * matrix(3, 3) - matrix(2, 3) * matrix(3, 0)) +
              matrix(1, 3) *
                (matrix(2, 0) * matrix(3, 1) - matrix(2, 1) * matrix(3, 0))) -
           matrix(0, 3) *
             (matrix(1, 0) *
                (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1)) -
              matrix(1, 1) *
                (matrix(2, 0) * matrix(3, 2) - matrix(2, 2) * matrix(3, 0)) +
              matrix(1, 2) *
                (matrix(2, 0) * matrix(3, 1) - matrix(2, 1) * matrix(3, 0)));
  }


  /**
   * Check if the vector is zero-valued by calculating its L1 norm.
   *
   * @param vec
   * @return
   */
  template <typename number>
  bool
  is_all_zero(const Vector<number> &vec)
  {
    if (vec.l1_norm() >
        std::numeric_limits<
          typename numbers::NumberTraits<number>::real_type>::epsilon())
      return false;
    else
      return true;
  }


  /**
   * Copy a segment of a vector into another vector at the specified location.
   *
   * @param dst_vec
   * @param dst_start_index
   * @param src_vec
   * @param start_index
   * @param number_of_data
   */
  template <typename number>
  void
  copy_vector(Vector<number>                          &dst_vec,
              const typename Vector<number>::size_type dst_start_index,
              const Vector<number>                    &src_vec,
              const typename Vector<number>::size_type src_start_index,
              const typename Vector<number>::size_type number_of_data)
  {
    std::memcpy((void *)(dst_vec.data() + dst_start_index),
                (void *)(src_vec.data() + src_start_index),
                number_of_data * sizeof(number));
  }


  /**
   * Compute the inner product of two vectors. When in the complex valued case,
   * complex conjugation is applied to the second operand.
   */
  template <typename VectorType1, typename VectorType2>
  typename VectorType1::value_type
  inner_product(const VectorType1 &vec1, const VectorType2 &vec2)
  {
    using Number1 = typename VectorType1::value_type;
    using Number2 = typename VectorType2::value_type;

    static_assert(is_number_larger_or_equal<Number1, Number2>());
    const size_type n = vec1.size();
    AssertDimension(n, vec2.size());

    if constexpr (std::is_same<Number1, Number2>::value)
      return BLASHelpers::inner_product_helper(n, vec1, vec2);
    else
      {
        Number1 result = Number1(0.);
        for (size_type i = 0; i < n; i++)
          result += vec1[i] *
                    Number1(numbers::NumberTraits<Number2>::conjugate(vec2[i]));

        return result;
      }
  }


  /**
   * Compute the linear combination of two vectors, which is different from the
   * inner product operation in that the second operand is not conjugated when
   * the number is complex valued.
   */
  template <typename VectorType1, typename VectorType2>
  typename VectorType1::value_type
  linear_combination(const VectorType1 &vec1, const VectorType2 &vec2)
  {
    using Number1 = typename VectorType1::value_type;
    using Number2 = typename VectorType2::value_type;

    static_assert(is_number_larger_or_equal<Number1, Number2>());
    const size_type n = vec1.size();
    AssertDimension(n, vec2.size());

    if constexpr (std::is_same<Number1, Number2>::value)
      return BLASHelpers::linear_combination_helper(n, vec1, vec2);
    else
      {
        Number1 result = Number1(0.);
        for (size_type i = 0; i < n; i++)
          result += vec1[i] * Number1(vec2[i]);

        return result;
      }
  }


  template <typename Number1, typename Number2>
  struct InnerProductOperation
  {
    static constexpr bool vectorizes = std::is_same<Number1, Number2>::value &&
                                       (std::is_same<Number1, float>::value ||
                                        std::is_same<Number1, double>::value) &&
                                       (VectorizedArray<Number1>::size() > 1);

    InnerProductOperation(const Number1 *const X, const Number2 *const Y)
      : X(X)
      , Y(Y)
    {
      static_assert(is_number_larger_or_equal<Number1, Number2>());
    }

    Number1
    operator()(const size_type i) const
    {
      return X[i] * Number1(numbers::NumberTraits<Number2>::conjugate(Y[i]));
    }

    VectorizedArray<Number1>
    do_vectorized(const size_type i) const
    {
      VectorizedArray<Number1> x, y;
      x.load(X + i);
      y.load(Y + i);

      // the following operation in VectorizedArray does an element-wise
      // scalar product without taking into account complex values and
      // the need to take the complex-conjugate of one argument. this
      // may be a bug, but because all VectorizedArray classes only
      // work on real scalars, it doesn't really matter very much.
      // in any case, assert that we really don't get here for
      // complex-valued objects
      static_assert(numbers::NumberTraits<Number1>::is_complex == false,
                    "This operation is not correctly implemented for "
                    "complex-valued objects.");
      return x * y;
    }

    const Number1 *const X;
    const Number2 *const Y;
  };


  template <typename Number1, typename Number2>
  struct LinearCombinationOperation
  {
    static constexpr bool vectorizes = std::is_same<Number1, Number2>::value &&
                                       (std::is_same<Number1, float>::value ||
                                        std::is_same<Number1, double>::value) &&
                                       (VectorizedArray<Number1>::size() > 1);

    LinearCombinationOperation(const Number1 *const X, const Number2 *const Y)
      : X(X)
      , Y(Y)
    {
      static_assert(is_number_larger_or_equal<Number1, Number2>());
    }

    Number1
    operator()(const size_type i) const
    {
      return X[i] * Number1(Y[i]);
    }

    VectorizedArray<Number1>
    do_vectorized(const size_type i) const
    {
      VectorizedArray<Number1> x, y;
      x.load(X + i);
      y.load(Y + i);

      // the following operation in VectorizedArray does an element-wise
      // scalar product without taking into account complex values and
      // the need to take the complex-conjugate of one argument. this
      // may be a bug, but because all VectorizedArray classes only
      // work on real scalars, it doesn't really matter very much.
      // in any case, assert that we really don't get here for
      // complex-valued objects
      static_assert(numbers::NumberTraits<Number1>::is_complex == false,
                    "This operation is not correctly implemented for "
                    "complex-valued objects.");
      return x * y;
    }

    const Number1 *const X;
    const Number2 *const Y;
  };


  /**
   * Compute the inner product of two vectors using TBB @p parallel_reduce . When
   * in the complex valued case, complex conjugation is applied to the second
   * operand.
   */
  template <typename VectorType1, typename VectorType2>
  typename VectorType1::value_type
  inner_product_tbb(const VectorType1 &vec1, const VectorType2 &vec2)
  {
    using Number1 = typename VectorType1::value_type;
    using Number2 = typename VectorType2::value_type;

    static_assert(is_number_larger_or_equal<Number1, Number2>());
    const size_type n = vec1.size();
    AssertDimension(n, vec2.size());

    Number1                                 result;
    InnerProductOperation<Number1, Number2> op(vec1.data(), vec2.data());
    auto partitioner = std::make_shared<parallel::internal::TBBPartitioner>();
    internal::VectorOperations::parallel_reduce(op, 0, n, result, partitioner);
    AssertIsFinite(result);

    return result;
  }


  /**
   * Compute the linear combination of two vectors using TBB @p parallel_reduce .
   */
  template <typename VectorType1, typename VectorType2>
  typename VectorType1::value_type
  linear_combination_tbb(const VectorType1 &vec1, const VectorType2 &vec2)
  {
    using Number1 = typename VectorType1::value_type;
    using Number2 = typename VectorType2::value_type;

    static_assert(is_number_larger_or_equal<Number1, Number2>());
    const size_type n = vec1.size();
    AssertDimension(n, vec2.size());

    Number1                                      result;
    LinearCombinationOperation<Number1, Number2> op(vec1.data(), vec2.data());
    auto partitioner = std::make_shared<parallel::internal::TBBPartitioner>();
    internal::VectorOperations::parallel_reduce(op, 0, n, result, partitioner);
    AssertIsFinite(result);

    return result;
  }


  template <typename Number1, typename Number2>
  struct Vector_real_part
  {
    Vector_real_part(const std::complex<Number2> *const src, Number1 *const dst)
      : src(src)
      , dst(dst)
    {
      static_assert(is_number_larger_or_equal<Number1, Number2>());
      static_assert(numbers::NumberTraits<Number1>::is_complex == false);

      Assert(src != nullptr, ExcInternalError());
      Assert(dst != nullptr, ExcInternalError());
    }

    void
    operator()(const size_type begin, const size_type end) const
    {
      Assert(end >= begin, ExcInternalError());

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (size_type i = begin; i < end; ++i)
        dst[i] = std::real(src[i]);
    }

    const std::complex<Number2> *const src;
    Number1 *const                     dst;
  };


  template <typename Number1, typename Number2>
  struct Vector_imag_part
  {
    Vector_imag_part(const std::complex<Number2> *const src, Number1 *const dst)
      : src(src)
      , dst(dst)
    {
      static_assert(is_number_larger_or_equal<Number1, Number2>());
      static_assert(numbers::NumberTraits<Number1>::is_complex == false);

      Assert(src != nullptr, ExcInternalError());
      Assert(dst != nullptr, ExcInternalError());
    }

    void
    operator()(const size_type begin, const size_type end) const
    {
      Assert(end >= begin, ExcInternalError());

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (size_type i = begin; i < end; ++i)
        dst[i] = std::imag(src[i]);
    }

    const std::complex<Number2> *const src;
    Number1 *const                     dst;
  };


  template <typename Number1, typename Number2>
  struct Vector_amplitude
  {
    Vector_amplitude(const std::complex<Number2> *const src, Number1 *const dst)
      : src(src)
      , dst(dst)
    {
      static_assert(is_number_larger_or_equal<Number1, Number2>());
      static_assert(numbers::NumberTraits<Number1>::is_complex == false);

      Assert(src != nullptr, ExcInternalError());
      Assert(dst != nullptr, ExcInternalError());
    }

    void
    operator()(const size_type begin, const size_type end) const
    {
      Assert(end >= begin, ExcInternalError());

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (size_type i = begin; i < end; ++i)
        dst[i] = std::abs(src[i]);
    }

    const std::complex<Number2> *const src;
    Number1 *const                     dst;
  };


  template <typename Number1, typename Number2>
  struct Vector_angle
  {
    Vector_angle(const std::complex<Number2> *const src, Number1 *const dst)
      : src(src)
      , dst(dst)
    {
      static_assert(is_number_larger_or_equal<Number1, Number2>());
      static_assert(numbers::NumberTraits<Number1>::is_complex == false);

      Assert(src != nullptr, ExcInternalError());
      Assert(dst != nullptr, ExcInternalError());
    }

    void
    operator()(const size_type begin, const size_type end) const
    {
      Assert(end >= begin, ExcInternalError());

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (size_type i = begin; i < end; ++i)
        dst[i] = std::arg(src[i]);
    }

    const std::complex<Number2> *const src;
    Number1 *const                     dst;
  };


  template <typename VectorType1, typename VectorType2>
  void
  get_vector_real_part(VectorType1 &vec1, const VectorType2 &vec2)
  {
    AssertDimension(vec1.size(), vec2.size());

    Vector_real_part op(vec2.data(), vec1.data());
    auto partitioner = std::make_shared<parallel::internal::TBBPartitioner>();
    dealii::internal::VectorOperations::parallel_for(op,
                                                     0,
                                                     vec1.size(),
                                                     partitioner);
  }


  template <typename VectorType1, typename VectorType2>
  void
  get_vector_imag_part(VectorType1 &vec1, const VectorType2 &vec2)
  {
    AssertDimension(vec1.size(), vec2.size());

    Vector_imag_part op(vec2.data(), vec1.data());
    auto partitioner = std::make_shared<parallel::internal::TBBPartitioner>();
    dealii::internal::VectorOperations::parallel_for(op,
                                                     0,
                                                     vec1.size(),
                                                     partitioner);
  }


  template <typename VectorType1, typename VectorType2>
  void
  get_vector_amplitude(VectorType1 &vec1, const VectorType2 &vec2)
  {
    AssertDimension(vec1.size(), vec2.size());

    Vector_amplitude op(vec2.data(), vec1.data());
    auto partitioner = std::make_shared<parallel::internal::TBBPartitioner>();
    dealii::internal::VectorOperations::parallel_for(op,
                                                     0,
                                                     vec1.size(),
                                                     partitioner);
  }


  template <typename VectorType1, typename VectorType2>
  void
  get_vector_angle(VectorType1 &vec1, const VectorType2 &vec2)
  {
    AssertDimension(vec1.size(), vec2.size());

    Vector_angle op(vec2.data(), vec1.data());
    auto partitioner = std::make_shared<parallel::internal::TBBPartitioner>();
    dealii::internal::VectorOperations::parallel_for(op,
                                                     0,
                                                     vec1.size(),
                                                     partitioner);
  }
} // namespace LinAlg

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_LINALG_H_
