/**
 * @file xp_utilities.h
 * @brief Introduction of xp_utilities.h
 *
 * @date 2023-02-19
 * @author Jihuan Tian
 */
#ifndef INCLUDE_UTILITIES_XP_H_
#define INCLUDE_UTILITIES_XP_H_

#include "config.h"

namespace HierBEM
{
  namespace Utilities
  {
    namespace CrossPlatform
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
    } // namespace CrossPlatform
  } // namespace Utilities
} // namespace HierBEM


#endif /* INCLUDE_UTILITIES_XP_H_ */
