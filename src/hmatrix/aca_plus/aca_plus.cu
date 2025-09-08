// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file aca_plus.cpp
 * @brief Introduction of aca_plus.cpp
 *
 * @date 2022-03-09
 * @author Jihuan Tian
 */

#include <random>

#include "config.h"
#include "hmatrix/aca_plus/aca_plus.hcu"

HBEM_NS_OPEN

// Global definition of the random number device and generator.
#if RANDOM_ACA
std::random_device rd;
#  ifdef DEAL_II_WITH_64BIT_INDICES
std::mt19937_64 rand_engine(rd());
#  else
std::mt19937 rand_engine(rd());
#  endif
#else
#  ifdef DEAL_II_WITH_64BIT_INDICES
std::mt19937_64 rand_engine;
#  else
std::mt19937 rand_engine;
#  endif
#endif

size_type
generate_random_index(const size_type a, const size_type b)
{
  std::uniform_int_distribution<size_type> uniform_distribution(a, b);

  return uniform_distribution(rand_engine);
}

HBEM_NS_CLOSE
