// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
// Copyright (C) 2024 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file utilities.h
 * @brief Introduction of utilities.h
 *
 * @date 2023-02-19
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_PLATFORM_SHARED_UTILITIES_H_
#define HIERBEM_INCLUDE_PLATFORM_SHARED_UTILITIES_H_

#include <deal.II/base/numbers.h>

#include <cmath>

#include "config.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

namespace PlatformShared
{
  namespace Utilities
  {
    template <int N, typename T>
    HBEM_ATTR_HOST HBEM_ATTR_DEV inline T
    fixed_power(const T x)
    {
      if constexpr (N == 0)
        return T(1.);
      else if constexpr (N < 0)
        return T(1.) / fixed_power<-N>(x);
      else if constexpr (N % 2 == 1)
        return x * fixed_power<N / 2>(x * x);
      else
        return fixed_power<N / 2>(x * x);
    }

    /**
     * Exponential function for real and complex values on the device.
     */
    template <typename Number>
    HBEM_ATTR_DEV inline Number
    exp(const Number x)
    {
      using real_type =
        typename dealii::numbers::NumberTraits<Number>::real_type;

      if constexpr (dealii::numbers::NumberTraits<Number>::is_complex)
        {
          // When the number is complex valued, we use the Euler's identity to
          // compute the exponential of the input number.
          const real_type e = ::exp(x.real());
          return Number(e * ::cos(x.imag()), e * ::sin(x.imag()));
        }
      else
        {
          // When the number is real valued, directly call the exponential
          // function.
          return ::exp(x);
        }
    }
  } // namespace Utilities
} // namespace PlatformShared

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PLATFORM_SHARED_UTILITIES_H_
