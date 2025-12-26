// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file helmholtz_acoustic_kernels.h
 * @brief Definition of kernel functions used in the Helmholtz acoustic
 * equation.
 *
 * @date 2025-12-13
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_PLATFORM_SHARED_HELMHOLTZ_ACOUSTIC_KERNELS_H_
#define HIERBEM_INCLUDE_PLATFORM_SHARED_HELMHOLTZ_ACOUSTIC_KERNELS_H_

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <assert.h>

#include <cmath>

#include "bem/types.h"
#include "config.h"
#include "platform_shared/tensor.h"
#include "platform_shared/utilities.h"

HBEM_NS_OPEN

using namespace dealii;

namespace PlatformShared
{
  namespace HelmholtzAcousticKernel
  {
    /**
     * Kernel function of the single layer potential integral operator, either
     * boundary integral operator or volume integral operator.
     */
    template <int spacedim, typename KernelNumberType>
    class SingleLayerKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType kernel_type;
      /**
       * Wave number. When it is a complex value, the medium is lossy. It is not
       * a const member, so that it can be negated or modified further.
       */
      KernelNumberType   kappa;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      SingleLayerKernel()
        : kernel_type(KernelType::SingleLayer)
        , kappa(0., 0.)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      SingleLayerKernel(const KernelNumberType kappa_)
        : kernel_type(KernelType::SingleLayer)
        , kappa(kappa_)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV void
      set_kappa(const KernelNumberType kappa_)
      {
        kappa = kappa_;
      }

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
            case 3: {
              const real_type r = (x - y).norm();
#ifdef __CUDA_ARCH__
              return (0.25 / numbers::PI / r *
                      ::exp(KernelNumberType(0., 1.) * kappa * r));
#else
              return (0.25 / numbers::PI / r *
                      std::exp(KernelNumberType(0., 1.) * kappa * r));
#endif
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the double layer potential integral operator, either
     * boundary integral operator or volume integral operator.
     */
    template <int spacedim, typename KernelNumberType>
    class DoubleLayerKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      KernelNumberType   kappa;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      DoubleLayerKernel()
        : kernel_type(KernelType::DoubleLayer)
        , kappa(0., 0.)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      DoubleLayerKernel(const KernelNumberType kappa_)
        : kernel_type(KernelType::DoubleLayer)
        , kappa(kappa_)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV void
      set_kappa(const KernelNumberType kappa_)
      {
        kappa = kappa_;
      }

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
            case 3: {
              const real_type        r   = (x - y).norm();
              const KernelNumberType ikr = KernelNumberType(0., 1.) * kappa * r;
#ifdef __CUDA_ARCH__
              return 0.25 / numbers::PI /
                     HierBEM::PlatformShared::Utilities::fixed_power<3>(r) *
                     ::exp(ikr) * (1 - ikr) * scalar_product(x - y, ny);
#else
              return 0.25 / numbers::PI /
                     HierBEM::PlatformShared::Utilities::fixed_power<3>(r) *
                     std::exp(ikr) * (1 - ikr) * scalar_product(x - y, ny);
#endif
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the adjoint double layer potential integral operator,
     * either boundary integral operator or volume integral operator.
     */
    template <int spacedim, typename KernelNumberType>
    class AdjointDoubleLayerKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      KernelNumberType   kappa;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      AdjointDoubleLayerKernel()
        : kernel_type(KernelType::AdjointDoubleLayer)
        , kappa(0., 0.)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      AdjointDoubleLayerKernel(const KernelNumberType kappa_)
        : kernel_type(KernelType::AdjointDoubleLayer)
        , kappa(kappa_)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV void
      set_kappa(const KernelNumberType kappa_)
      {
        kappa = kappa_;
      }

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
            case 3: {
              const real_type        r   = (x - y).norm();
              const KernelNumberType ikr = KernelNumberType(0., 1.) * kappa * r;
#ifdef __CUDA_ARCH__
              return 0.25 / numbers::PI /
                     HierBEM::PlatformShared::Utilities::fixed_power<3>(r) *
                     ::exp(ikr) * (1 - ikr) * scalar_product(y - x, nx);
#else
              return 0.25 / numbers::PI /
                     HierBEM::PlatformShared::Utilities::fixed_power<3>(r) *
                     std::exp(ikr) * (1 - ikr) * scalar_product(y - x, nx);
#endif
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the hypersingular volume integral operator.
     */
    template <int spacedim, typename KernelNumberType>
    class HyperSingularKernel
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      KernelNumberType   kappa;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernel()
        : kernel_type(KernelType::HyperSingular)
        , kappa(0., 0.)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernel(const KernelNumberType kappa_)
        : kernel_type(KernelType::HyperSingular)
        , kappa(kappa_)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV void
      set_kappa(const KernelNumberType kappa_)
      {
        kappa = kappa_;
      }

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

      switch (spacedim)
        {
            case 3: {
              const real_type        r   = (x - y).norm();
              const real_type        r2  = r * r;
              const real_type        r3  = r * r2;
              const real_type        r5  = r2 * r3;
              const KernelNumberType ikr = KernelNumberType(0., 1.) * kappa * r;

#ifdef __CUDA_ARCH__
              return 0.25 / numbers::PI * ::exp(ikr) *
                     ((ikr - 1.) * scalar_product(nx, ny) / r3 -
                      (kappa * kappa / r3 + 3 * (ikr - 1.) / r5) *
                        scalar_product(x - y, nx) * scalar_product(x - y, ny));
#else
              return 0.25 / numbers::PI * std::exp(ikr) *
                     ((ikr - 1.) * scalar_product(nx, ny) / r3 -
                      (kappa * kappa / r3 + 3 * (ikr - 1.) / r5) *
                        scalar_product(x - y, nx) * scalar_product(x - y, ny));
#endif
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the hypersingular boundary integral operator, which
     * requires regularization. This class is the first part in this kernel
     * function, which requires surface curl to be applied to the trial and test
     * functions.
     */
    template <int spacedim, typename KernelNumberType>
    class HyperSingularKernelRegular1
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      KernelNumberType   kappa;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernelRegular1()
        : kernel_type(KernelType::HyperSingularRegular)
        , kappa(0., 0.)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernelRegular1(const KernelNumberType kappa_)
        : kernel_type(KernelType::HyperSingularRegular)
        , kappa(kappa_)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV void
      set_kappa(const KernelNumberType kappa_)
      {
        kappa = kappa_;
      }

      /**
       * Calculate the value of fundamental solution of the Helmholtz operator,
       * which is also the kernel function of the single layer potential
       * integral operator. This version runs on the CPU host.
       *
       * \mynote{Because regularization will be applied to the bilinear form
       * of the hyper-singular kernel, the value calculated here is actually
       * not the hyper-singular function itself, but the fundamental solution
       * of the Helmholtz operator.}
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
        return false;
      }
    };


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_HOST HBEM_ATTR_DEV KernelNumberType
    HyperSingularKernelRegular1<spacedim, KernelNumberType>::value(
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
            case 3: {
              const real_type r = (x - y).norm();
#ifdef __CUDA_ARCH__
              return (0.25 / numbers::PI / r *
                      ::exp(KernelNumberType(0., 1.) * kappa * r));
#else
              return (0.25 / numbers::PI / r *
                      std::exp(KernelNumberType(0., 1.) * kappa * r));
#endif
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    /**
     * Kernel function of the hypersingular boundary integral operator, which
     * requires regularization. This class is the second part in this kernel
     * function, which does not require surface curl to be applied to the trial
     * and test functions.
     */
    template <int spacedim, typename KernelNumberType>
    class HyperSingularKernelRegular2
    {
    public:
      using real_type =
        typename numbers::NumberTraits<KernelNumberType>::real_type;
      static constexpr unsigned int dimension = spacedim;

      const KernelType   kernel_type;
      KernelNumberType   kappa;
      const unsigned int n_components;

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernelRegular2()
        : kernel_type(KernelType::HyperSingularRegular)
        , kappa(0., 0.)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV
      HyperSingularKernelRegular2(const KernelNumberType kappa_)
        : kernel_type(KernelType::HyperSingularRegular)
        , kappa(kappa_)
        , n_components(1)
      {}

      HBEM_ATTR_HOST HBEM_ATTR_DEV void
      set_kappa(const KernelNumberType kappa_)
      {
        kappa = kappa_;
      }

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
    HyperSingularKernelRegular2<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const
    {
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();
#ifdef __CUDA_ARCH__
              return (-kappa * kappa * 0.25 / numbers::PI / r *
                      ::exp(KernelNumberType(0., 1.) * kappa * r) *
                      scalar_product(nx, ny));
#else
              return (-kappa * kappa * 0.25 / numbers::PI / r *
                      std::exp(KernelNumberType(0., 1.) * kappa * r) *
                      scalar_product(nx, ny));
#endif
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }
  } // namespace HelmholtzAcousticKernel
} // namespace PlatformShared

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PLATFORM_SHARED_HELMHOLTZ_ACOUSTIC_KERNELS_H_