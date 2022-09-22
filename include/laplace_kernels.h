/**
 * @file laplace_kernels.h
 * @brief Introduction of laplace_kernels.h
 *
 * @date 2022-03-04
 * @author Jihuan Tian
 */
#ifndef INCLUDE_LAPLACE_KERNELS_H_
#define INCLUDE_LAPLACE_KERNELS_H_

#include "bem_kernels.h"

using namespace IdeoBEM;

namespace LaplaceKernel
{
  /**
   * Laplace single layer kernel function
   */
  template <int dim, typename RangeNumberType = double>
  class SingleLayerKernel : public KernelFunction<dim, RangeNumberType>
  {
  public:
    SingleLayerKernel()
      : KernelFunction<dim, RangeNumberType>(SingleLayer)
    {}

    /**
     * Evaluate the kernel function.
     *
     * \mynote{With the appended keyword @p override, this function must be
     * explicitly defined.}
     *
     * @param x
     * @param y
     * @param nx
     * @param ny
     * @param component
     * @return
     */
    virtual RangeNumberType
    value(const Point<dim> &    x,
          const Point<dim> &    y,
          const Tensor<1, dim> &nx,
          const Tensor<1, dim> &ny,
          const unsigned int    component = 0) const override;

    /**
     * Return whether the kernel function is symmetric.
     *
     * @return
     */
    virtual bool
    is_symmetric() const override;
  };


  template <int dim, typename RangeNumberType>
  RangeNumberType
  SingleLayerKernel<dim, RangeNumberType>::value(
    const Point<dim> &    x,
    const Point<dim> &    y,
    const Tensor<1, dim> &nx,
    const Tensor<1, dim> &ny,
    const unsigned int    component) const
  {
    (void)nx;
    (void)ny;
    (void)component;

    switch (dim)
      {
        case 2:
          return (-0.5 / numbers::PI * std::log(1.0 / (x - y).norm()));

        case 3:
          return (0.25 / numbers::PI / (x - y).norm());

        default:
          Assert(false, ExcInternalError());
          return 0.;
      }
  }


  template <int dim, typename RangeNumberType>
  bool
  SingleLayerKernel<dim, RangeNumberType>::is_symmetric() const
  {
    return true;
  }


  /**
   * Double layer kernel.
   */
  template <int dim, typename RangeNumberType = double>
  class DoubleLayerKernel : public KernelFunction<dim, RangeNumberType>
  {
  public:
    DoubleLayerKernel()
      : KernelFunction<dim, RangeNumberType>(DoubleLayer)
    {}

    virtual RangeNumberType
    value(const Point<dim> &    x,
          const Point<dim> &    y,
          const Tensor<1, dim> &nx,
          const Tensor<1, dim> &ny,
          const unsigned int    component = 0) const override;

    /**
     * Return whether the kernel function is symmetric.
     *
     * @return
     */
    virtual bool
    is_symmetric() const override;
  };


  template <int dim, typename RangeNumberType>
  RangeNumberType
  DoubleLayerKernel<dim, RangeNumberType>::value(
    const Point<dim> &    x,
    const Point<dim> &    y,
    const Tensor<1, dim> &nx,
    const Tensor<1, dim> &ny,
    const unsigned int    component) const
  {
    (void)nx;
    (void)component;

    switch (dim)
      {
        case 2:
          return ((y - x) * ny) / 2.0 / numbers::PI / (y - x).norm_square();

        case 3:
          return ((x - y) * ny) / 4.0 / numbers::PI /
                 Utilities::fixed_power<3>((x - y).norm());

        default:
          Assert(false, ExcInternalError());
          return 0.;
      }
  }


  template <int dim, typename RangeNumberType>
  bool
  DoubleLayerKernel<dim, RangeNumberType>::is_symmetric() const
  {
    return false;
  }


  // Class for the adjoint double layer kernel.
  template <int dim, typename RangeNumberType = double>
  class AdjointDoubleLayerKernel : public KernelFunction<dim, RangeNumberType>
  {
  public:
    AdjointDoubleLayerKernel()
      : KernelFunction<dim, RangeNumberType>(AdjointDoubleLayer)
    {}

    virtual RangeNumberType
    value(const Point<dim> &    x,
          const Point<dim> &    y,
          const Tensor<1, dim> &nx,
          const Tensor<1, dim> &ny,
          const unsigned int    component = 0) const override;

    /**
     * Return whether the kernel function is symmetric.
     *
     * @return
     */
    virtual bool
    is_symmetric() const override;
  };


  template <int dim, typename RangeNumberType>
  RangeNumberType
  AdjointDoubleLayerKernel<dim, RangeNumberType>::value(
    const Point<dim> &    x,
    const Point<dim> &    y,
    const Tensor<1, dim> &nx,
    const Tensor<1, dim> &ny,
    const unsigned int    component) const
  {
    (void)ny;
    (void)component;

    switch (dim)
      {
        case 2:
          return ((x - y) * nx) / 2.0 / numbers::PI / (x - y).norm_square();

        case 3:
          return ((y - x) * nx) / 4.0 / numbers::PI /
                 Utilities::fixed_power<3>((x - y).norm());

        default:
          Assert(false, ExcInternalError());
          return 0.;
      }
  }


  template <int dim, typename RangeNumberType>
  bool
  AdjointDoubleLayerKernel<dim, RangeNumberType>::is_symmetric() const
  {
    return false;
  }


  /**
   * Kernel function for the hyper singular boundary integral operator.
   */
  template <int dim, typename RangeNumberType = double>
  class HyperSingularKernel : public KernelFunction<dim, RangeNumberType>
  {
  public:
    HyperSingularKernel()
      : KernelFunction<dim, RangeNumberType>(HyperSingular)
    {}

    virtual RangeNumberType
    value(const Point<dim> &    x,
          const Point<dim> &    y,
          const Tensor<1, dim> &nx,
          const Tensor<1, dim> &ny,
          const unsigned int    component = 0) const override;

    /**
     * Return whether the kernel function is symmetric.
     *
     * @return
     */
    virtual bool
    is_symmetric() const override;
  };


  template <int dim, typename RangeNumberType>
  RangeNumberType
  HyperSingularKernel<dim, RangeNumberType>::value(
    const Point<dim> &    x,
    const Point<dim> &    y,
    const Tensor<1, dim> &nx,
    const Tensor<1, dim> &ny,
    const unsigned int    component) const
  {
    (void)component;

    double r2 = (x - y).norm_square();

    switch (dim)
      {
        case 2:
          {
            double r4 = r2 * r2;

            return 0.5 / numbers::PI *
                   (-nx * ny / r2 + 2.0 * (nx * (x - y)) * (ny * (x - y)) / r4);
          }

        case 3:
          {
            double r3 = (x - y).norm() * r2;
            double r5 = r2 * r3;

            return 0.25 / numbers::PI *
                   (-nx * ny / r3 + 3.0 * (nx * (x - y)) * (ny * (x - y)) / r5);
          }

        default:
          {
            Assert(false, ExcInternalError());
            return 0.;
          }
      }
  }

  template <int dim, typename RangeNumberType>
  bool
  HyperSingularKernel<dim, RangeNumberType>::is_symmetric() const
  {
    return true;
  }


  /**
   * Kernel function for the regularized hyper singular boundary integral
   * operator.
   */
  template <int dim, typename RangeNumberType = double>
  class HyperSingularKernelRegular : public KernelFunction<dim, RangeNumberType>
  {
  public:
    HyperSingularKernelRegular()
      : KernelFunction<dim, RangeNumberType>(HyperSingularRegular)
    {}

    virtual RangeNumberType
    value(const Point<dim> &    x,
          const Point<dim> &    y,
          const Tensor<1, dim> &nx,
          const Tensor<1, dim> &ny,
          const unsigned int    component = 0) const override;

    /**
     * Return whether the kernel function is symmetric.
     *
     * @return
     */
    virtual bool
    is_symmetric() const override;
  };


  template <int dim, typename RangeNumberType>
  RangeNumberType
  HyperSingularKernelRegular<dim, RangeNumberType>::value(
    const Point<dim> &    x,
    const Point<dim> &    y,
    const Tensor<1, dim> &nx,
    const Tensor<1, dim> &ny,
    const unsigned int    component) const
  {
    (void)nx;
    (void)ny;
    (void)component;

    switch (dim)
      {
        case 2:
          {
            return (-0.5 / numbers::PI * std::log((x - y).norm()));
          }

        case 3:
          {
            return (0.25 / numbers::PI / (x - y).norm());
          }

        default:
          {
            Assert(false, ExcInternalError());
            return 0.;
          }
      }
  }

  template <int dim, typename RangeNumberType>
  bool
  HyperSingularKernelRegular<dim, RangeNumberType>::is_symmetric() const
  {
    return true;
  }
} // namespace LaplaceKernel

#endif /* INCLUDE_LAPLACE_KERNELS_H_ */
