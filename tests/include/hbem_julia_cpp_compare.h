/**
 * @file hbem_julia_cpp_compare.h
 * @brief Compare C++ results with Julia.
 *
 * @date 2025-03-08
 * @author Jihuan Tian
 */

#ifndef HIERBEM_TESTS_INCLUDE_HBEM_JULIA_CPP_COMPARE_H_
#define HIERBEM_TESTS_INCLUDE_HBEM_JULIA_CPP_COMPARE_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <complex>
#include <functional>
#include <type_traits>

#include "config.h"
#include "hbem_julia_wrapper.h"
#include "linear_algebra/lapack_full_matrix_ext.h"

HBEM_NS_OPEN

using namespace Catch::Matchers;
using namespace dealii;

/**
 * Compare with a scalar value. N.B. When a complex value is to be compared
 * with, we should compare with its real and imaginary parts separately.
 */
template <typename Number>
void
compare_with_jl_scalar(
  const Number                                           value,
  const std::string                                     &jl_value_name,
  const double                                           abs_error = 1e-15,
  const double                                           rel_error = 1e-15,
  const std::function<typename numbers::NumberTraits<Number>::real_type(
    typename numbers::NumberTraits<Number>::real_type)> &func =
    [](typename numbers::NumberTraits<Number>::real_type v) ->
  typename numbers::NumberTraits<Number>::real_type { return v; })
{
  HBEMJuliaWrapper &inst     = HBEMJuliaWrapper::get_instance();
  HBEMJuliaValue    jl_value = inst.eval_string(jl_value_name);

  if constexpr (std::is_same<Number, int>::value)
    {
      REQUIRE_THAT(func(value),
                   WithinAbs(func(jl_value.int_value()), abs_error) ||
                     WithinRel(func(jl_value.int_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, unsigned int>::value)
    {
      REQUIRE_THAT(func(value),
                   WithinAbs(func(jl_value.uint_value()), abs_error) ||
                     WithinRel(func(jl_value.uint_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, float>::value)
    {
      REQUIRE_THAT(func(value),
                   WithinAbs(func(jl_value.float_value()), abs_error) ||
                     WithinRel(func(jl_value.float_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, double>::value)
    {
      REQUIRE_THAT(func(value),
                   WithinAbs(func(jl_value.double_value()), abs_error) ||
                     WithinRel(func(jl_value.double_value()), rel_error));
    }
}


template <typename Number>
void
compare_with_jl_complex(
  const std::complex<Number>                             value,
  const std::string                                     &jl_value_name,
  const double                                           abs_error = 1e-15,
  const double                                           rel_error = 1e-15,
  const std::function<typename numbers::NumberTraits<Number>::real_type(
    typename numbers::NumberTraits<Number>::real_type)> &func =
    [](typename numbers::NumberTraits<Number>::real_type v) ->
  typename numbers::NumberTraits<Number>::real_type { return v; })
{
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();

  // Compare the real part.
  HBEMJuliaValue jl_value =
    inst.eval_string(std::string("real(") + jl_value_name + std::string(")"));

  if constexpr (std::is_same<Number, int>::value)
    {
      REQUIRE_THAT(func(value.real()),
                   WithinAbs(func(jl_value.int_value()), abs_error) ||
                     WithinRel(func(jl_value.int_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, unsigned int>::value)
    {
      REQUIRE_THAT(func(value.real()),
                   WithinAbs(func(jl_value.uint_value()), abs_error) ||
                     WithinRel(func(jl_value.uint_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, float>::value)
    {
      REQUIRE_THAT(func(value.real()),
                   WithinAbs(func(jl_value.float_value()), abs_error) ||
                     WithinRel(func(jl_value.float_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, double>::value)
    {
      REQUIRE_THAT(func(value.real()),
                   WithinAbs(func(jl_value.double_value()), abs_error) ||
                     WithinRel(func(jl_value.double_value()), rel_error));
    }

  // Compare the imaginary part
  jl_value =
    inst.eval_string(std::string("imag(") + jl_value_name + std::string(")"));

  if constexpr (std::is_same<Number, int>::value)
    {
      REQUIRE_THAT(func(value.imag()),
                   WithinAbs(func(jl_value.int_value()), abs_error) ||
                     WithinRel(func(jl_value.int_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, unsigned int>::value)
    {
      REQUIRE_THAT(func(value.imag()),
                   WithinAbs(func(jl_value.uint_value()), abs_error) ||
                     WithinRel(func(jl_value.uint_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, float>::value)
    {
      REQUIRE_THAT(func(value.imag()),
                   WithinAbs(func(jl_value.float_value()), abs_error) ||
                     WithinRel(func(jl_value.float_value()), rel_error));
    }
  else if constexpr (std::is_same<Number, double>::value)
    {
      REQUIRE_THAT(func(value.imag()),
                   WithinAbs(func(jl_value.double_value()), abs_error) ||
                     WithinRel(func(jl_value.double_value()), rel_error));
    }
}


template <typename Number, template <typename> typename VectorType>
void
compare_with_jl_array(
  const VectorType<Number>                              &array,
  const std::string                                     &jl_array_name,
  const double                                           abs_error = 1e-15,
  const double                                           rel_error = 1e-15,
  const std::function<typename numbers::NumberTraits<Number>::real_type(
    typename numbers::NumberTraits<Number>::real_type)> &func =
    [](typename numbers::NumberTraits<Number>::real_type v) ->
  typename numbers::NumberTraits<Number>::real_type { return v; })
{
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();

  HBEMJuliaValue array_jl      = inst.eval_string(jl_array_name);
  Number        *array_jl_data = array_jl.array<Number>();
  const size_t   n             = array.size();
  REQUIRE(array_jl.nrows() == n);

  for (size_t i = 0; i < n; i++)
    {
      if constexpr (numbers::NumberTraits<Number>::is_complex)
        {
          REQUIRE_THAT(func(array[i].real()),
                       WithinAbs(func(array_jl_data[i].real()), abs_error) ||
                         WithinRel(func(array_jl_data[i].real()), rel_error));
          REQUIRE_THAT(func(array[i].imag()),
                       WithinAbs(func(array_jl_data[i].imag()), abs_error) ||
                         WithinRel(func(array_jl_data[i].imag()), rel_error));
        }
      else
        {
          REQUIRE_THAT(func(array[i]),
                       WithinAbs(func(array_jl_data[i]), abs_error) ||
                         WithinRel(func(array_jl_data[i]), rel_error));
        }
    }
}


template <typename Number>
void
compare_with_jl_array(
  const LAPACKFullMatrixExt<Number>                     &mat,
  const std::string                                     &jl_array_name,
  const double                                           abs_error = 1e-15,
  const double                                           rel_error = 1e-15,
  const std::function<typename numbers::NumberTraits<Number>::real_type(
    typename numbers::NumberTraits<Number>::real_type)> &func =
    [](typename numbers::NumberTraits<Number>::real_type v) ->
  typename numbers::NumberTraits<Number>::real_type { return v; })
{
  const size_t n = std::max(mat.m(), mat.n());
  AssertDimension(std::min(mat.m(), mat.n()), 1);

  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();

  HBEMJuliaValue array_jl      = inst.eval_string(jl_array_name);
  Number        *array_jl_data = array_jl.array<Number>();
  REQUIRE(array_jl.nrows() == n);

  for (size_t i = 0; i < n; i++)
    {
      if constexpr (numbers::NumberTraits<Number>::is_complex)
        {
          REQUIRE_THAT(func(mat.m() == 1 ? mat(0, i).real() : mat(i, 0).real()),
                       WithinAbs(func(array_jl_data[i].real()), abs_error) ||
                         WithinRel(func(array_jl_data[i].real()), rel_error));
          REQUIRE_THAT(func(mat.m() == 1 ? mat(0, i).imag() : mat(i, 0).imag()),
                       WithinAbs(func(array_jl_data[i].imag()), abs_error) ||
                         WithinRel(func(array_jl_data[i].imag()), rel_error));
        }
      else
        {
          REQUIRE_THAT(func(mat.m() == 1 ? mat(0, i) : mat(i, 0)),
                       WithinAbs(func(array_jl_data[i]), abs_error) ||
                         WithinRel(func(array_jl_data[i]), rel_error));
        }
    }
}


template <typename Number, template <typename> typename MatrixType>
void
compare_with_jl_matrix(
  const MatrixType<Number>                              &mat,
  const std::string                                     &jl_mat_name,
  const double                                           abs_error = 1e-15,
  const double                                           rel_error = 1e-15,
  const std::function<typename numbers::NumberTraits<Number>::real_type(
    typename numbers::NumberTraits<Number>::real_type)> &func =
    [](typename numbers::NumberTraits<Number>::real_type v) ->
  typename numbers::NumberTraits<Number>::real_type { return v; })
{
  HBEMJuliaWrapper &inst = HBEMJuliaWrapper::get_instance();

  HBEMJuliaValue mat_jl      = inst.eval_string(jl_mat_name);
  Number        *mat_jl_data = mat_jl.array<Number>();
  const size_t   m           = mat.m();
  const size_t   n           = mat.n();
  REQUIRE(mat_jl.size(0) == m);
  REQUIRE(mat_jl.size(1) == n);

  for (size_t j = 0; j < n; j++)
    for (size_t i = 0; i < m; i++)
      {
        if constexpr (numbers::NumberTraits<Number>::is_complex)
          {
            REQUIRE_THAT(
              func(mat(i, j).real()),
              WithinAbs(func(mat_jl_data[i + j * m].real()), abs_error) ||
                WithinRel(func(mat_jl_data[i + j * m].real()), rel_error));
            REQUIRE_THAT(
              func(mat(i, j).imag()),
              WithinAbs(func(mat_jl_data[i + j * m].imag()), abs_error) ||
                WithinRel(func(mat_jl_data[i + j * m].imag()), rel_error));
          }
        else
          {
            REQUIRE_THAT(func(mat(i, j)),
                         WithinAbs(func(mat_jl_data[i + j * m]), abs_error) ||
                           WithinRel(func(mat_jl_data[i + j * m]), rel_error));
          }
      }
}

HBEM_NS_CLOSE

#endif
