/**
 * @file number_traits.h
 * @brief Definition of ranks for different types of numbers
 * @ingroup
 *
 * @date 2025-04-08
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_NUMBER_TRAITS_H_
#define HIERBEM_INCLUDE_NUMBER_TRAITS_H_

#include <deal.II/base/numbers.h>

#include <complex>
#include <type_traits>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Struct for assigning rank values to fundamental number types.
 */
template <typename Number>
struct type_rank;

template <>
struct type_rank<float> : std::integral_constant<int, 1>
{};

template <>
struct type_rank<double> : std::integral_constant<int, 2>
{};

template <>
struct type_rank<std::complex<float>> : std::integral_constant<int, 3>
{};

template <>
struct type_rank<std::complex<double>> : std::integral_constant<int, 4>
{};


/**
 * Whether Number1 and Number2 are comparable.
 */
template <typename Number1, typename Number2>
constexpr bool
is_number_comparable()
{
  if constexpr (numbers::NumberTraits<Number1>::is_complex &&
                numbers::NumberTraits<Number2>::is_complex)
    /* Both numbers are complex. */
    return true;
  if constexpr (!numbers::NumberTraits<Number1>::is_complex &&
                !numbers::NumberTraits<Number2>::is_complex)
    /* Both numbers are real. */
    return true;
  else if constexpr (numbers::NumberTraits<Number1>::is_complex)
    /* Number1 is complex and Number2 is real. */
    return type_rank<
             typename numbers::NumberTraits<Number1>::real_type>::value >=
           type_rank<Number2>::value;
  else
    /* Number2 is complex and Number1 is real. */
    return type_rank<
             typename numbers::NumberTraits<Number2>::real_type>::value >=
           type_rank<Number1>::value;
}


/**
 * Whether Number1 and Number2 are comparable and Number1 > Number2.
 */
template <typename Number1, typename Number2>
constexpr bool
is_number_larger()
{
  return is_number_comparable<Number1, Number2>() &&
         type_rank<Number1>::value > type_rank<Number2>::value;
}


/**
 * Whether Number1 and Number2 are comparable and Number1 >= Number2.
 */
template <typename Number1, typename Number2>
constexpr bool
is_number_larger_or_equal()
{
  return is_number_comparable<Number1, Number2>() &&
         type_rank<Number1>::value >= type_rank<Number2>::value;
}


/**
 * Whether Number1 and Number2 are comparable and Number1 < Number2.
 */
template <typename Number1, typename Number2>
constexpr bool
is_number_smaller()
{
  return is_number_comparable<Number1, Number2>() &&
         type_rank<Number1>::value < type_rank<Number2>::value;
}


/**
 * Whether Number1 and Number2 are comparable and Number1 <= Number2.
 */
template <typename Number1, typename Number2>
constexpr bool
is_number_smaller_or_equal()
{
  return is_number_comparable<Number1, Number2>() &&
         type_rank<Number1>::value <= type_rank<Number2>::value;
}

HBEM_NS_CLOSE

#endif /* HIERBEM_INCLUDE_NUMBER_TRAITS_H_ */
