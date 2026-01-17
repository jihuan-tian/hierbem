// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file concepts.h
 * @brief Definition of concepts.
 *
 * @date 2026-01-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_UTILITIES_CONCEPTS_H_
#define HIERBEM_INCLUDE_UTILITIES_CONCEPTS_H_

#include <deal.II/base/numbers.h>

#include <complex>

#include "config.h"
#include "cuda_complex.hcu"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

template <typename KernelNumberType>
concept HostComplex = std::is_same_v<
  KernelNumberType,
  std::complex<typename numbers::NumberTraits<KernelNumberType>::real_type>>;

template <typename KernelNumberType>
concept DeviceComplex =
  std::is_same_v<KernelNumberType,
                 HierBEM::complex<typename numbers::NumberTraits<
                   KernelNumberType>::real_type>>;

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_UTILITIES_CONCEPTS_H_
