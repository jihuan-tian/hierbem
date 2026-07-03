// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
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
 * @file laplace_kernels.h
 * @brief Introduction of laplace_kernels.h
 *
 * @date 2022-03-04
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_PLATFORM_SHARED_LAPLACE_KERNELS_H_
#define HIERBEM_INCLUDE_PLATFORM_SHARED_LAPLACE_KERNELS_H_

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cassert>

#include "bem/types.h"
#include "config.h"
#include "platform_shared/tensor.h"
#include "platform_shared/utilities.h"

HBEM_NS_OPEN

using namespace dealii;

namespace PlatformShared
{
  namespace LaplaceKernel
  {
    /**
     * Kernel function of the single layer potential integral operator, either
     * boundary integral operator or volume integral operator.
     */
    template <int spacedim, typename KernelNumberType = double>
    class SingleLayerKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      SingleLayerKernel()
        : kernel_type(KernelType::SingleLayer)
        , n_components(1)
      {}

      /**
       * Evaluate the kernel function.
       *
       * @param x
       * @param y
       * @param nx
       * @param ny
       * @param component
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const;

      /**
       * Return whether the kernel function is symmetric.
       *
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      is_symmetric() const
      {
        return true;
      }

      /**
       * Whether regularization is needed when source points and target points
       * may overlap.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_regularization() const
      {
        return false;
      }

      /**
       * Whether stabilization is needed when the operator is defined on the
       * full domain. If needed, it means the operator lacks ellipticity.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_stabilization_on_full_domain() const
      {
        return false;
      }
    };


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
    SingleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const
    {
      (void)nx;
      (void)ny;
      (void)component;

      switch (spacedim)
        {
          case 2:
#ifdef __CUDA_ARCH__
            return (-0.5 / numbers::PI * ::log(1.0 / (x - y).norm()));
#else
            return (-0.5 / numbers::PI * std::log(1.0 / (x - y).norm()));
#endif

          case 3:
            return (0.25 / numbers::PI / (x - y).norm());

          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the double layer potential integral operator, either
     * boundary integral operator or volume integral operator.
     */
    template <int spacedim, typename KernelNumberType = double>
    class DoubleLayerKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      DoubleLayerKernel()
        : kernel_type(KernelType::DoubleLayer)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const;

      /**
       * Return whether the kernel function is symmetric.
       *
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      is_symmetric() const
      {
        return false;
      }

      /**
       * Whether regularization is needed when source points and target points
       * may overlap.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_regularization() const
      {
        return false;
      }

      /**
       * Whether stabilization is needed when the operator is defined on the
       * full domain. If needed, it means the operator lacks ellipticity.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_stabilization_on_full_domain() const
      {
        return false;
      }
    };


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
    DoubleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const
    {
      (void)nx;
      (void)component;

      switch (spacedim)
        {
          case 2:
            return scalar_product((y - x), ny) / 2.0 / numbers::PI /
                   (y - x).norm_square();

          case 3:
            return scalar_product((x - y), ny) / 4.0 / numbers::PI /
                   HierBEM::PlatformShared::Utilities::fixed_power<3>(
                     (x - y).norm());

          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the adjoint double layer potential integral operator,
     * either boundary integral operator or volume integral operator.
     */
    template <int spacedim, typename KernelNumberType = double>
    class AdjointDoubleLayerKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      AdjointDoubleLayerKernel()
        : kernel_type(KernelType::AdjointDoubleLayer)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const;

      /**
       * Return whether the kernel function is symmetric.
       *
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      is_symmetric() const
      {
        return false;
      }

      /**
       * Whether regularization is needed when source points and target points
       * may overlap.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_regularization() const
      {
        return false;
      }

      /**
       * Whether stabilization is needed when the operator is defined on the
       * full domain. If needed, it means the operator lacks ellipticity.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_stabilization_on_full_domain() const
      {
        return false;
      }
    };


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
    AdjointDoubleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const
    {
      (void)ny;
      (void)component;

      switch (spacedim)
        {
          case 2:
            return scalar_product((x - y), nx) / 2.0 / numbers::PI /
                   (x - y).norm_square();

          case 3:
            return scalar_product((y - x), nx) / 4.0 / numbers::PI /
                   HierBEM::PlatformShared::Utilities::fixed_power<3>(
                     (x - y).norm());

          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the hypersingular volume integral operator.
     */
    template <int spacedim, typename KernelNumberType = double>
    class HyperSingularKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernel()
        : kernel_type(KernelType::HyperSingular)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const;

      /**
       * Return whether the kernel function is symmetric.
       *
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      is_symmetric() const
      {
        return true;
      }

      /**
       * Whether regularization is needed when source points and target points
       * may overlap.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_regularization() const
      {
        return false;
      }

      /**
       * Whether stabilization is needed when the operator is defined on the
       * full domain. If needed, it means the operator lacks ellipticity.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_stabilization_on_full_domain() const
      {
        return false;
      }
    };


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
    HyperSingularKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const
    {
      (void)component;

      const real_type r2 = (x - y).norm_square();

      switch (spacedim)
        {
            case 2: {
              const real_type r4 = r2 * r2;

              return 0.5 / numbers::PI *
                     (-scalar_product(nx, ny) / r2 +
                      2.0 * scalar_product(nx, (x - y)) *
                        scalar_product(ny, (x - y)) / r4);
            }

            case 3: {
              const real_type r3 = (x - y).norm() * r2;
              const real_type r5 = r2 * r3;

              return 0.25 / numbers::PI *
                     (-scalar_product(nx, ny) / r3 +
                      3.0 * scalar_product(nx, (x - y)) *
                        scalar_product(ny, (x - y)) / r5);
            }

          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the hypersingular boundary integral operator, which
     * requires regularization, i.e. surface curl should be be applied to the
     * trial and test functions.
     */
    template <int spacedim, typename KernelNumberType = double>
    class HyperSingularKernelRegular
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernelRegular()
        : kernel_type(KernelType::HyperSingularRegular)
        , n_components(1)
      {}

      /**
       * Calculate the value of fundamental solution of the Laplace operator.
       * This version runs on the CPU host.
       *
       * \mynote{Because regularization will be applied to the bilinear form
       * of the hyper-singular kernel, the value calculated here is actually
       * not the hyper-singular function itself, but the fundamental solution
       * of the Laplace operator. The final computing the regularized bilinear
       * form will be carried out in the pullback in the unit cell, which is
       * handled in the
       * function @p KernelPulledbackToUnitCell::value.}
       *
       * @param x
       * @param y
       * @param nx
       * @param ny
       * @param component
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const;

      /**
       * Return whether the kernel function is symmetric.
       *
       * @return
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      is_symmetric() const
      {
        return true;
      }

      /**
       * Whether regularization is needed when source points and target points
       * may overlap.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_regularization() const
      {
        return true;
      }

      /**
       * Whether stabilization is needed when the operator is defined on the
       * full domain. If needed, it means the operator lacks ellipticity.
       */
      HBEM_ATTR_HOST HBEM_ATTR_DEV bool
      needs_stabilization_on_full_domain() const
      {
        return true;
      }
    };


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
    HyperSingularKernelRegular<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const
    {
      (void)nx;
      (void)ny;
      (void)component;

      switch (spacedim)
        {
          case 2:
#ifdef __CUDA_ARCH__
            return (-0.5 / numbers::PI * ::log((x - y).norm()));
#else
            return (-0.5 / numbers::PI * std::log((x - y).norm()));
#endif

          case 3:
            return (0.25 / numbers::PI / (x - y).norm());

          default:
            assert(false);
            return KernelNumberType();
        }
    }
  } // namespace LaplaceKernel
} // namespace PlatformShared

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PLATFORM_SHARED_LAPLACE_KERNELS_H_
