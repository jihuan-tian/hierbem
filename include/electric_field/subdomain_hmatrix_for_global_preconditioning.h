// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file subdomain_hmatrix_for_global_preconditioning.h
 * @brief Introduction of subdomain_hmatrix_for_global_preconditioning.h
 *
 * @date 2024-08-08
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_GLOBAL_PRECONDITIONING_H_
#define HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_GLOBAL_PRECONDITIONING_H_

#include "config.h"
#include "subdomain_steklov_poincare_hmatrix.h"

HBEM_NS_OPEN

template <int spacedim, typename Number = double>
class SubdomainHMatrixForGlobalPreconditioning
{
public:
  /**
   * Calculate \hmatrix/vector multiplication as \f$y = y + M \cdot x\f$.
   *
   * @param y Result vector
   * @param x Input vector
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;

  /**
   * Calculate \hmatrix/vector multiplication as \f$y = y + \alpha \cdot M
   * \cdot x\f$.
   *
   * @param y Result vector
   * @param alpha Scalar factor before \f$x\f$
   * @param x Input vector
   */
  void
  vmult(Vector<Number> &y, const Number alpha, const Vector<Number> &x) const;

private:
  SubdomainSteklovPoincareHMatrix<spacedim, Number> *S;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_ELECTRIC_FIELD_SUBDOMAIN_HMATRIX_FOR_GLOBAL_PRECONDITIONING_H_
