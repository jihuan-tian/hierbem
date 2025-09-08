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

#include "config.h"

HBEM_NS_OPEN

namespace PlatformShared
{
  namespace Utilities
  {
    template <int N, typename T>
    HBEM_ATTR_HOST HBEM_ATTR_DEV inline T
    fixed_power(const T x)
    {
      if (N == 0)
        return T(1.);
      else if (N < 0)
        return T(1.) / fixed_power<-N>(x);
      else
        // Use exponentiation by squaring:
        return ((N % 2 == 1) ? x * fixed_power<N / 2>(x * x) :
                               fixed_power<N / 2>(x * x));
    }
  } // namespace Utilities
} // namespace PlatformShared

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PLATFORM_SHARED_UTILITIES_H_
