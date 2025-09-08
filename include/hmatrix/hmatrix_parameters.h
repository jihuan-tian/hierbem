// Copyright (C) -2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_INCLUDE_HMATRIX_HMATRIX_PARAMETERS_H_
#define HIERBEM_INCLUDE_HMATRIX_HMATRIX_PARAMETERS_H_

#include "config.h"

HBEM_NS_OPEN

/**
 * @brief Class holding the set of \hmatrix parameters.
 */
template <typename Number = double>
class HMatrixParameters
{
public:
  HMatrixParameters(const unsigned int n_min_for_ct,
                    const unsigned int n_min_for_bct,
                    const Number       eta,
                    const unsigned int max_hmat_rank,
                    const Number       aca_relative_error)
    : n_min_for_ct(n_min_for_ct)
    , n_min_for_bct(n_min_for_bct)
    , eta(eta)
    , max_hmat_rank(max_hmat_rank)
    , aca_relative_error(aca_relative_error)
  {}

  /**
   * Minimum cluster node size in the cluster tree.
   */
  unsigned int n_min_for_ct;
  /**
   * Minimum block cluster node size in the block cluster tree.
   */
  unsigned int n_min_for_bct;
  /**
   * Admissibility constant.
   */
  Number eta;
  /**
   * Maximum \hmat rank.
   */
  unsigned int max_hmat_rank;
  /**
   * Relative error for ACA iteration.
   */
  Number aca_relative_error;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_HMATRIX_PARAMETERS_H_
