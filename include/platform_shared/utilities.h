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
