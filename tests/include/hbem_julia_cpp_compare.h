/**
 * @file hbem_julia_cpp_compare.h
 * @brief Compare C++ results with Julia.
 *
 * @date 2025-03-08
 * @author Jihuan Tian
 */

#ifndef HIERBEM_TESTS_INCLUDE_HBEM_JULIA_CPP_COMPARE_H_
#define HIERBEM_TESTS_INCLUDE_HBEM_JULIA_CPP_COMPARE_H_

#include <deal.II/base/numbers.h>

#include <catch2/catch_all.hpp>

#include "config.h"
#include "hbem_julia_wrapper.h"

HBEM_NS_OPEN

using namespace Catch::Matchers;
using namespace dealii;

template <typename Number, template <typename> typename VectorType>
void
compare_with_jl_array(
  const VectorType<Number>                               &array,
  const std::string                                       jl_array_name,
  const typename numbers::NumberTraits<Number>::real_type abs_error = 1e-15,
  const typename numbers::NumberTraits<Number>::real_type rel_error = 1e-15)
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
          REQUIRE_THAT(array[i].real(),
                       WithinAbs(array_jl_data[i].real(), abs_error) ||
                         WithinRel(array_jl_data[i].real(), rel_error));
          REQUIRE_THAT(array[i].imag(),
                       WithinAbs(array_jl_data[i].imag(), abs_error) ||
                         WithinRel(array_jl_data[i].imag(), rel_error));
        }
      else
        {
          REQUIRE_THAT(array[i],
                       WithinAbs(array_jl_data[i], abs_error) ||
                         WithinRel(array_jl_data[i], rel_error));
        }
    }
}


template <typename Number, template <typename> typename MatrixType>
void
compare_with_jl_matrix(
  const MatrixType<Number>                               &mat,
  const std::string                                       jl_mat_name,
  const typename numbers::NumberTraits<Number>::real_type abs_error = 1e-15,
  const typename numbers::NumberTraits<Number>::real_type rel_error = 1e-15)
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
            REQUIRE_THAT(mat(i, j).real(),
                         WithinAbs(mat_jl_data[i + j * m].real(), abs_error) ||
                           WithinRel(mat_jl_data[i + j * m].real(), rel_error));
            REQUIRE_THAT(mat(i, j).imag(),
                         WithinAbs(mat_jl_data[i + j * m].imag(), abs_error) ||
                           WithinRel(mat_jl_data[i + j * m].imag(), rel_error));
          }
        else
          {
            REQUIRE_THAT(mat(i, j),
                         WithinAbs(mat_jl_data[i + j * m], abs_error) ||
                           WithinRel(mat_jl_data[i + j * m], rel_error));
          }
      }
}

HBEM_NS_CLOSE

#endif
