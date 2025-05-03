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
#include "cuda_complex.hcu"

DEAL_II_NAMESPACE_OPEN

namespace numbers
{
  /**
   * complex is compatible with CUDA.
   */
  template <typename Number>
  struct is_cuda_compatible<HierBEM::complex<Number>, void> : std::true_type
  {};


  /**
   * Specialization of the general NumberTraits class that provides the
   * relevant information if the underlying data type is complex<T>.
   */
  template <typename number>
  struct NumberTraits<HierBEM::complex<number>>
  {
    /**
     * A flag that specifies whether the template type given to this class is
     * complex or real. Since this specialization of the general template is
     * selected for complex types, the answer is <code>true</code>.
     */
    static constexpr bool is_complex = true;

    /**
     * For this data type, alias the corresponding real type. Since this
     * specialization of the template is selected for number types
     * complex<T>, the real type is equal to the type used to store the
     * two components of the complex number.
     */
    using real_type = number;

    /**
     * For this data type, alias the corresponding double type.
     */
    using double_type = HierBEM::complex<double>;

    /**
     * Return the complex-conjugate of the given number.
     */
    static constexpr HierBEM::complex<number>
    conjugate(const HierBEM::complex<number> &x);

    /**
     * Return the square of the absolute value of the given number. Since this
     * specialization of the general template is chosen for types equal to
     * complex, this function returns the product of a number and its
     * complex conjugate.
     */
    static constexpr real_type
    abs_square(const HierBEM::complex<number> &x);


    /**
     * Return the absolute value of a complex number.
     */
    static real_type
    abs(const HierBEM::complex<number> &x);
  };


  inline bool
  is_finite(const HierBEM::complex<double> &x)
  {
    // Check complex numbers for infinity
    // by testing real and imaginary part
    return (is_finite(x.real()) && is_finite(x.imag()));
  }


  inline bool
  is_finite(const HierBEM::complex<float> &x)
  {
    // Check complex numbers for infinity
    // by testing real and imaginary part
    return (is_finite(x.real()) && is_finite(x.imag()));
  }


  template <typename number>
  constexpr HBEM_ATTR_HOST HBEM_ATTR_DEV HierBEM::complex<number>
  NumberTraits<HierBEM::complex<number>>::conjugate(
    const HierBEM::complex<number> &x)
  {
    return conj(x);
  }


  template <typename number>
  HBEM_ATTR_HOST HBEM_ATTR_DEV
    typename NumberTraits<HierBEM::complex<number>>::real_type
    NumberTraits<HierBEM::complex<number>>::abs(
      const HierBEM::complex<number> &x)
  {
    return abs(x);
  }


  template <typename number>
  constexpr HBEM_ATTR_HOST HBEM_ATTR_DEV
    typename NumberTraits<HierBEM::complex<number>>::real_type
    NumberTraits<HierBEM::complex<number>>::abs_square(
      const HierBEM::complex<number> &x)
  {
    return norm(x);
  }
} // namespace numbers

DEAL_II_NAMESPACE_CLOSE

HBEM_NS_OPEN

using namespace dealii;

/**
 * Define an alias type on the device which is the counter part of the host
 * number type.
 */
template <typename HostNumberType>
using DeviceNumberType = std::conditional_t<
  std::is_floating_point_v<HostNumberType>,
  HostNumberType,
  complex<typename numbers::NumberTraits<HostNumberType>::real_type>>;

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

template <>
struct type_rank<complex<float>> : std::integral_constant<int, 3>
{};

template <>
struct type_rank<complex<double>> : std::integral_constant<int, 4>
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
