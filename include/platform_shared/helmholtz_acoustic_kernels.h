// Copyright (C) 2025-2026 Jihuan Tian <jihuan_tian@hotmail.com>
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
#include <complex>

#include "bem/types.h"
#include "config.h"
#include "platform_shared/tensor.h"
#include "platform_shared/utilities.h"
#include "utilities/concepts.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

namespace PlatformShared
{
  namespace HelmholtzAcousticKernel
  {
    /**
     * Kernel function of the single layer potential integral operator,
     * either boundary integral operator or volume integral operator.
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

      HBEM_ATTR_HOST
      SingleLayerKernel() requires HostComplex<KernelNumberType>
        : kernel_type(KernelType::SingleLayer), kappa(0., 0.), n_components(1)
      {}

      HBEM_ATTR_DEV
      SingleLayerKernel() requires DeviceComplex<KernelNumberType>
        : kernel_type(KernelType::SingleLayer), kappa(0., 0.), n_components(1)
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
      HBEM_ATTR_HOST KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        HostComplex<KernelNumberType>;

      HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        DeviceComplex<KernelNumberType>;

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
    HBEM_ATTR_HOST KernelNumberType
    SingleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int component) const requires HostComplex<KernelNumberType>
    {
      (void)nx;
      (void)ny;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();

              return (real_type(0.25 / numbers::PI) / r *
                      std::exp(KernelNumberType(0., 1.0) * kappa * r));
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_DEV KernelNumberType
    SingleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const requires
      DeviceComplex<KernelNumberType>
    {
      (void)nx;
      (void)ny;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();

              return (real_type(0.25 / numbers::PI) / r *
                      PlatformShared::Utilities::exp(KernelNumberType(0., 1.0) *
                                                     kappa * r));
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

      HBEM_ATTR_HOST
      DoubleLayerKernel() requires HostComplex<KernelNumberType>
        : kernel_type(KernelType::DoubleLayer), kappa(0., 0.), n_components(1)
      {}

      HBEM_ATTR_DEV
      DoubleLayerKernel() requires DeviceComplex<KernelNumberType>
        : kernel_type(KernelType::DoubleLayer), kappa(0., 0.), n_components(1)
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

      HBEM_ATTR_HOST KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        HostComplex<KernelNumberType>;

      HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        DeviceComplex<KernelNumberType>;

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
    HBEM_ATTR_HOST KernelNumberType
    DoubleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int component) const requires HostComplex<KernelNumberType>
    {
      (void)nx;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type        r = (x - y).norm();
              const KernelNumberType ikr =
                KernelNumberType(0., 1.0) * kappa * r;

              return real_type(0.25 / numbers::PI) /
                     PlatformShared::Utilities::fixed_power<3>(r) *
                     std::exp(ikr) * (real_type(1.0) - ikr) *
                     scalar_product(x - y, ny);
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_DEV KernelNumberType
    DoubleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const requires
      DeviceComplex<KernelNumberType>
    {
      (void)nx;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type        r = (x - y).norm();
              const KernelNumberType ikr =
                KernelNumberType(0., 1.0) * kappa * r;

              return real_type(0.25 / numbers::PI) /
                     PlatformShared::Utilities::fixed_power<3>(r) *
                     PlatformShared::Utilities::exp(ikr) *
                     (real_type(1.0) - ikr) * scalar_product(x - y, ny);
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

      HBEM_ATTR_HOST
      AdjointDoubleLayerKernel() requires HostComplex<KernelNumberType>
        : kernel_type(KernelType::AdjointDoubleLayer),
          kappa(0., 0.),
          n_components(1)
      {}

      HBEM_ATTR_DEV
      AdjointDoubleLayerKernel() requires DeviceComplex<KernelNumberType>
        : kernel_type(KernelType::AdjointDoubleLayer),
          kappa(0., 0.),
          n_components(1)
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

      HBEM_ATTR_HOST KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        HostComplex<KernelNumberType>;

      HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        DeviceComplex<KernelNumberType>;

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
    HBEM_ATTR_HOST KernelNumberType
    AdjointDoubleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int component) const requires HostComplex<KernelNumberType>
    {
      (void)ny;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type        r = (x - y).norm();
              const KernelNumberType ikr =
                KernelNumberType(0., 1.0) * kappa * r;

              return real_type(0.25 / numbers::PI) /
                     PlatformShared::Utilities::fixed_power<3>(r) *
                     std::exp(ikr) * (real_type(1.0) - ikr) *
                     scalar_product(y - x, nx);
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_DEV KernelNumberType
    AdjointDoubleLayerKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const requires
      DeviceComplex<KernelNumberType>
    {
      (void)ny;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type        r = (x - y).norm();
              const KernelNumberType ikr =
                KernelNumberType(0., 1.0) * kappa * r;

              return real_type(0.25 / numbers::PI) /
                     PlatformShared::Utilities::fixed_power<3>(r) *
                     PlatformShared::Utilities::exp(ikr) *
                     (real_type(1.0) - ikr) * scalar_product(y - x, nx);
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

      HBEM_ATTR_HOST
      HyperSingularKernel() requires HostComplex<KernelNumberType>
        : kernel_type(KernelType::HyperSingular), kappa(0., 0.), n_components(1)
      {}

      HBEM_ATTR_DEV
      HyperSingularKernel() requires DeviceComplex<KernelNumberType>
        : kernel_type(KernelType::HyperSingular), kappa(0., 0.), n_components(1)
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

      HBEM_ATTR_HOST KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        HostComplex<KernelNumberType>;

      HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        DeviceComplex<KernelNumberType>;

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
    HBEM_ATTR_HOST KernelNumberType
    HyperSingularKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int component) const requires HostComplex<KernelNumberType>
    {
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type        r  = (x - y).norm();
              const real_type        r2 = r * r;
              const real_type        r3 = r * r2;
              const real_type        r5 = r2 * r3;
              const KernelNumberType ikr =
                KernelNumberType(0., 1.0) * kappa * r;

              return real_type(0.25 / numbers::PI) * std::exp(ikr) *
                     ((ikr - real_type(1.0)) * scalar_product(nx, ny) / r3 -
                      (kappa * kappa / r3 +
                       real_type(3.0) * (ikr - real_type(1.0)) / r5) *
                        scalar_product(x - y, nx) * scalar_product(x - y, ny));
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_DEV KernelNumberType
    HyperSingularKernel<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const requires
      DeviceComplex<KernelNumberType>
    {
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type        r  = (x - y).norm();
              const real_type        r2 = r * r;
              const real_type        r3 = r * r2;
              const real_type        r5 = r2 * r3;
              const KernelNumberType ikr =
                KernelNumberType(0., 1.0) * kappa * r;

              return real_type(0.25 / numbers::PI) *
                     PlatformShared::Utilities::exp(ikr) *
                     ((ikr - real_type(1.0)) * scalar_product(nx, ny) / r3 -
                      (kappa * kappa / r3 +
                       real_type(3.0) * (ikr - real_type(1.0)) / r5) *
                        scalar_product(x - y, nx) * scalar_product(x - y, ny));
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

      HBEM_ATTR_HOST
      HyperSingularKernelRegular1() requires HostComplex<KernelNumberType>
        : kernel_type(KernelType::HyperSingularRegular),
          kappa(0., 0.),
          n_components(1)
      {}

      HBEM_ATTR_DEV
      HyperSingularKernelRegular1() requires DeviceComplex<KernelNumberType>
        : kernel_type(KernelType::HyperSingularRegular),
          kappa(0., 0.),
          n_components(1)
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
      HBEM_ATTR_HOST KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        HostComplex<KernelNumberType>;

      HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        DeviceComplex<KernelNumberType>;

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
    HBEM_ATTR_HOST KernelNumberType
    HyperSingularKernelRegular1<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int component) const requires HostComplex<KernelNumberType>
    {
      (void)nx;
      (void)ny;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();

              return (real_type(0.25 / numbers::PI) / r *
                      std::exp(KernelNumberType(0., 1.0) * kappa * r));
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_DEV KernelNumberType
    HyperSingularKernelRegular1<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const requires
      DeviceComplex<KernelNumberType>
    {
      (void)nx;
      (void)ny;
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();

              return (real_type(0.25 / numbers::PI) / r *
                      PlatformShared::Utilities::exp(KernelNumberType(0., 1.0) *
                                                     kappa * r));
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

      HBEM_ATTR_HOST
      HyperSingularKernelRegular2() requires HostComplex<KernelNumberType>
        : kernel_type(KernelType::HyperSingularRegular),
          kappa(0., 0.),
          n_components(1)
      {}

      HBEM_ATTR_DEV
      HyperSingularKernelRegular2() requires DeviceComplex<KernelNumberType>
        : kernel_type(KernelType::HyperSingularRegular),
          kappa(0., 0.),
          n_components(1)
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

      HBEM_ATTR_HOST KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        HostComplex<KernelNumberType>;

      HBEM_ATTR_DEV KernelNumberType
      value(const Point<spacedim, real_type>     &x,
            const Point<spacedim, real_type>     &y,
            const Tensor<1, spacedim, real_type> &nx,
            const Tensor<1, spacedim, real_type> &ny,
            const unsigned int                    component = 0) const requires
        DeviceComplex<KernelNumberType>;

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
    HBEM_ATTR_HOST KernelNumberType
    HyperSingularKernelRegular2<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int component) const requires HostComplex<KernelNumberType>
    {
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();

              return (-kappa * kappa * real_type(0.25 / numbers::PI) / r *
                      std::exp(KernelNumberType(0., 1.0) * kappa * r) *
                      scalar_product(nx, ny));
            }
          default:
            assert(false);
            return KernelNumberType();
        }
    }


    template <int spacedim, typename KernelNumberType>
    HBEM_ATTR_DEV KernelNumberType
    HyperSingularKernelRegular2<spacedim, KernelNumberType>::value(
      const Point<spacedim, real_type>     &x,
      const Point<spacedim, real_type>     &y,
      const Tensor<1, spacedim, real_type> &nx,
      const Tensor<1, spacedim, real_type> &ny,
      const unsigned int                    component) const requires
      DeviceComplex<KernelNumberType>
    {
      (void)component;

      switch (spacedim)
        {
            case 3: {
              const real_type r = (x - y).norm();

              return (-kappa * kappa * real_type(0.25 / numbers::PI) / r *
                      PlatformShared::Utilities::exp(KernelNumberType(0., 1.0) *
                                                     kappa * r) *
                      scalar_product(nx, ny));
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
