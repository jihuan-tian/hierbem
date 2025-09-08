// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_CONFIG_H_
#define HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_CONFIG_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * Configuration for ACA+.
 */
template <typename Number = double>
struct ACAConfig
{
  ACAConfig()
    : max_iter(0)
    , epsilon(0.)
    , eta(0.)
  {}

  ACAConfig(unsigned int v_max_iter, Number v_epsilon, Number v_eta)
    : max_iter(v_max_iter)
    , epsilon(v_epsilon)
    , eta(v_eta)
  {}

  /**
   * Maximum number of iteration, which is also the maximum rank \f$k\f$ for
   * the far field matrix block to be built.
   */
  unsigned int max_iter;
  /**
   * Relative error between the current cross and the approximant matrix
   * \f$S\f$, i.e. \f[ \norm{u_k}_2\norm{v_k}_2 \leq
   * \frac{\varepsilon(1-\eta)}{1+\varepsilon} \norm{S}_{\rm F}. \f]
   */
  Number epsilon;
  /**
   * Admissibility constant
   */
  Number eta;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_CONFIG_H_
