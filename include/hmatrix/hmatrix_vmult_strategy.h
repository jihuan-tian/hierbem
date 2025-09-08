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
 * @file hmatrix_vmult_strategy.h
 * @brief Strategies for top level H-matrix/vector multiplications, namely,
 * @p vmult , @p Tvmult , @p Hvmult , @p vmult_add , @p Tvmult_add and
 * @p Hvmult_add .
 * @ingroup hierarchical_matrices
 *
 * @date 2025-08-28
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_HMATRIX_HMATRIX_VMULT_STRATEGY_H_
#define HIERBEM_INCLUDE_HMATRIX_HMATRIX_VMULT_STRATEGY_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>

#include <deal.II/lac/vector.h>

#include <complex>
#include <memory>

#include "config.h"
#include "hmatrix/hmatrix_support.h"

HBEM_NS_OPEN

using namespace dealii;

template <int spacedim, typename Number>
class HMatrix;

/**
 * Abstract base class for top level H-matrix/vector multiplication strategies.
 */
template <int spacedim, typename Number>
class HMatrixVmultStrategy
{
public:
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  virtual ~HMatrixVmultStrategy() = default;

  virtual void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const = 0;

  virtual void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const = 0;

  virtual void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const = 0;

  virtual void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;


  virtual void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;


  virtual void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;
};


// Partial specialization of the class @p HMatrixVmultStrategy .
template <int spacedim, typename T>
class HMatrixVmultStrategy<spacedim, std::complex<T>>
{
public:
  using Number    = std::complex<T>;
  using real_type = T;

  virtual ~HMatrixVmultStrategy() = default;

  virtual void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const = 0;

  virtual void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const = 0;

  virtual void
  vmult(Vector<Number>                  &y,
        const Number                     alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const = 0;

  virtual void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const = 0;

  virtual void
  vmult_add(Vector<Number>                  &y,
            const Number                     alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const = 0;

  virtual void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;

  virtual void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const = 0;
};


/**
 * Class for top level H-matrix/vector multiplication using the serial recursive
 * strategy.
 */
template <int spacedim, typename Number>
class HMatrixVmultSerialRecursive
  : public HMatrixVmultStrategy<spacedim, Number>
{
public:
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.vmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.vmult(y, alpha, x, hmat.get_property());
  }

  void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult(y, alpha, x, hmat.get_property());
  }

  void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Tvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Tvmult(y, alpha, x, hmat.get_property());
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult(y, alpha, x, hmat.get_property());
  }

  void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Hvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Hvmult(y, alpha, x, hmat.get_property());
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult(y, alpha, x, hmat.get_property());
  }
};


// Partial specialization of the class @p HMatrixVmultSerialRecursive .
template <int spacedim, typename T>
class HMatrixVmultSerialRecursive<spacedim, std::complex<T>>
  : public HMatrixVmultStrategy<spacedim, std::complex<T>>
{
public:
  using Number    = std::complex<T>;
  using real_type = T;

  void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.vmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  vmult(Vector<Number>                  &y,
        const Number                     alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.vmult(y, alpha, x, hmat.get_property());
  }

  void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.vmult(y, alpha, x, hmat.get_property());
  }

  void
  vmult_add(Vector<Number>                  &y,
            const Number                     alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult(y, alpha, x, hmat.get_property());
  }

  void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult(y, alpha, x, hmat.get_property());
  }

  void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Tvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Tvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Tvmult(y, alpha, x, hmat.get_property());
  }

  void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Tvmult(y, alpha, x, hmat.get_property());
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult(y, alpha, x, hmat.get_property());
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult(y, alpha, x, hmat.get_property());
  }

  void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Hvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult(y, real_type(1.0), x, hmat.get_property());
  }

  void
  Hvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Hvmult(y, alpha, x, hmat.get_property());
  }

  void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    y = Number(0.);
    hmat.Hvmult(y, alpha, x, hmat.get_property());
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult(y, alpha, x, hmat.get_property());
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult(y, alpha, x, hmat.get_property());
  }
};


/**
 * Class for top level H-matrix/vector multiplication using the serial iterative
 * strategy.
 */
template <int spacedim, typename Number>
class HMatrixVmultSerialIterative
  : public HMatrixVmultStrategy<spacedim, Number>
{
public:
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(0.), y, real_type(1.0), x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(1.0), y, real_type(1.0), x);
  }

  void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(0.), y, real_type(1.0), x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(0.), y, real_type(1.0), x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(1.0), y, alpha, x);
  }
};


// Partial specialization of the class @p HMatrixVmultSerialIterative .
template <int spacedim, typename T>
class HMatrixVmultSerialIterative<spacedim, std::complex<T>>
  : public HMatrixVmultStrategy<spacedim, std::complex<T>>
{
public:
  using Number    = std::complex<T>;
  using real_type = T;

  void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(0.), y, real_type(1.0), x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(1.0), y, real_type(1.0), x);
  }

  void
  vmult(Vector<Number>                  &y,
        const Number                     alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const Number                     alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(0.), y, real_type(1.0), x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(0.), y, real_type(1.0), x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(0.), y, alpha, x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(1.0), y, alpha, x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_serial_iterative(real_type(1.0), y, alpha, x);
  }
};


/**
 * Class for top level H-matrix/vector multiplication using the TBB task
 * parallel strategy.
 */
template <int spacedim, typename Number>
class HMatrixVmultTaskParallel : public HMatrixVmultStrategy<spacedim, Number>
{
public:
  using real_type = typename numbers::NumberTraits<Number>::real_type;

  void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(0.), y, real_type(1.0), x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(1.0), y, real_type(1.0), x);
  }

  void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(0.), y, real_type(1.0), x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(0.), y, real_type(1.0), x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(1.0), y, alpha, x);
  }
};


// Partial specialization of the class @p HMatrixVmultTaskParallel .
template <int spacedim, typename T>
class HMatrixVmultTaskParallel<spacedim, std::complex<T>>
  : public HMatrixVmultStrategy<spacedim, std::complex<T>>
{
public:
  using Number    = std::complex<T>;
  using real_type = T;

  void
  vmult(Vector<Number>                  &y,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(0.), y, real_type(1.0), x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(1.0), y, real_type(1.0), x);
  }

  void
  vmult(Vector<Number>                  &y,
        const Number                     alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  vmult(Vector<Number>                  &y,
        const real_type                  alpha,
        const HMatrix<spacedim, Number> &hmat,
        const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const Number                     alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  vmult_add(Vector<Number>                  &y,
            const real_type                  alpha,
            const HMatrix<spacedim, Number> &hmat,
            const Vector<Number>            &x) const override
  {
    hmat.vmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(0.), y, real_type(1.0), x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  Tvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  Tvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Tvmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(0.), y, real_type(1.0), x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(1.0), y, real_type(1.0), x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const Number                     alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  Hvmult(Vector<Number>                  &y,
         const real_type                  alpha,
         const HMatrix<spacedim, Number> &hmat,
         const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(0.), y, alpha, x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const Number                     alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(1.0), y, alpha, x);
  }

  void
  Hvmult_add(Vector<Number>                  &y,
             const real_type                  alpha,
             const HMatrix<spacedim, Number> &hmat,
             const Vector<Number>            &x) const override
  {
    hmat.Hvmult_task_parallel(real_type(1.0), y, alpha, x);
  }
};


/**
 * @brief Enum for the strategy adopted by top level H-matrix @p vmult ,
 * @p Tvmult or @p Hvmult functions, which are called by iterative solvers.
 */
enum IterativeSolverVmultType
{
  SerialRecursive,
  SerialIterative,
  TaskParallel
};

inline const char *
vmult_type_name(const IterativeSolverVmultType type)
{
  switch (type)
    {
      case IterativeSolverVmultType::SerialRecursive:
        return "serial-recursive";
      case IterativeSolverVmultType::SerialIterative:
        return "serial-iterative";
      case IterativeSolverVmultType::TaskParallel:
        return "task-parallel";
      default:
        return "unknown";
    }
}

/**
 * Set the strategy adopted by top level H-matrix/vector multiplication related
 * functions, namely, @p vmult , @p Tvmult and @p Hvmult , as well as
 * @p vmult_add , @p Tvmult_add and @p Hvmult_add .
 *
 * When the strategy is task parallel, the two flags @p is_vmult and @p is_tvmult
 * will be used to prepare necessary data on each thread.
 */
template <int spacedim, typename Number>
void
set_hmatrix_vmult_strategy_for_iterative_solver(
  HMatrix<spacedim, Number>     &hmat,
  const IterativeSolverVmultType type,
  const bool                     is_vmult  = true,
  const bool                     is_tvmult = false)
{
  Assert(hmat.is_root(), ExcInternalError());

  switch (type)
    {
        case IterativeSolverVmultType::SerialRecursive: {
          hmat.set_vmult_strategy(
            std::make_unique<HMatrixVmultSerialRecursive<spacedim, Number>>());

          break;
        }
        case IterativeSolverVmultType::SerialIterative: {
          hmat.set_vmult_strategy(
            std::make_unique<HMatrixVmultSerialIterative<spacedim, Number>>());

          break;
        }
        case IterativeSolverVmultType::TaskParallel: {
          hmat.set_vmult_strategy(
            std::make_unique<HMatrixVmultTaskParallel<spacedim, Number>>());
          hmat.prepare_for_vmult_or_tvmult(is_vmult, is_tvmult);

          break;
        }
        default: {
          Assert(false, ExcInternalError());
          break;
        }
    }
}

HBEM_NS_CLOSE

#endif /* HIERBEM_INCLUDE_HMATRIX_HMATRIX_VMULT_STRATEGY_H_ */
