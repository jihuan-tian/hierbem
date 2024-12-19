/**
 * @file xp_tensor.h
 * @brief Introduction of xp_tensor.h
 *
 * @date 2023-02-10
 * @author Jihuan Tian
 */
#ifndef INCLUDE_TENSOR_XP_H_
#define INCLUDE_TENSOR_XP_H_

#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>

#include "config.h"

namespace HierBEM
{
  using namespace dealii;

  namespace CrossPlatform
  {
    template <int dim, typename Number1, typename Number2>
    HBEM_ATTR_HOST
      HBEM_ATTR_DEV Tensor<1, dim, typename ProductType<Number1, Number2>::type>
                    cross_product_3d(const Tensor<1, dim, Number1> &src1,
                                     const Tensor<1, dim, Number2> &src2)
    {
      Tensor<1, dim, typename ProductType<Number1, Number2>::type> result;

      // avoid compiler warnings
      constexpr int s0 = 0 % dim;
      constexpr int s1 = 1 % dim;
      constexpr int s2 = 2 % dim;

      result[s0] = src1[s1] * src2[s2] - src1[s2] * src2[s1];
      result[s1] = src1[s2] * src2[s0] - src1[s0] * src2[s2];
      result[s2] = src1[s0] * src2[s1] - src1[s1] * src2[s0];

      return result;
    }

    /**
     * Calculate the scalar product of two rank-1 tensors.
     *
     * @tparam dim
     * @tparam Number1
     * @tparam Number2
     * @param t1
     * @param t2
     * @return
     */
    template <int dim, typename Number1, typename Number2>
    HBEM_ATTR_HOST HBEM_ATTR_DEV typename ProductType<Number1, Number2>::type
                   scalar_product(const Tensor<1, dim, Number1> &t1,
                                  const Tensor<1, dim, Number2> &t2)
    {
      typename ProductType<Number1, Number2>::type result = 0.;

#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
      for (unsigned int i = 0; i < dim; i++)
        {
          result += t1[i] * t2[i];
        }

      return result;
    }
  } // namespace CrossPlatform
} // namespace HierBEM


#endif /* INCLUDE_TENSOR_XP_H_ */
