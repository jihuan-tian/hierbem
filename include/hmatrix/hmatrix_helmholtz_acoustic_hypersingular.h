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
 * @file hmatrix_helmholtz_acoustic_hypersingular.h
 * @brief Definition of the H-matrix for the hypersingular boundary integral
 * operator used in Helmholtz acoustic equation.
 *
 * @date 2025-12-14
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_HMATRIX_HMATRIX_HELMHOLTZ_ACOUSTIC_HYPERSINGULAR_H_
#define HIERBEM_INCLUDE_HMATRIX_HMATRIX_HELMHOLTZ_ACOUSTIC_HYPERSINGULAR_H_

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/vector.h>

#include "config.h"
#include "hmatrix/hmatrix.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Class for the H-matrix associated with the regularized hypersingular boundary
 * integral operator \f$D_{\kappa}\f$ used in the Helmholtz acoustic equation.
 */
template <int spacedim, typename Number>
class HMatrixHelmholtzAcousticHypersingular
{
public:
  HMatrixHelmholtzAcousticHypersingular(HMatrix<spacedim, Number> &D1_,
                                        HMatrix<spacedim, Number> &D2_)
    : D1(D1_)
    , D2(D2_)
  {}

  /**
   * Calculte the \hmat/vector multiplication \f$y = D_{\kappa} \cdot x = D_1
   * \cdot x + D_2 \cdot x\f$.
   *
   * @param y
   * @param x
   */
  void
  vmult(Vector<Number> &y, const Vector<Number> &x) const;

private:
  /**
   * First part of the H-matrix for the operator \f$D_{\kappa}\f$, which
   * involves surface curl of trial and test functions.
   */
  HMatrix<spacedim, Number> &D1;
  /**
   * Second part of the H-matrix for the operator \f$D_{\kappa}\f$.
   */
  HMatrix<spacedim, Number> &D2;
};


template <int spacedim, typename Number>
void
HMatrixHelmholtzAcousticHypersingular<spacedim, Number>::vmult(
  Vector<Number>       &y,
  const Vector<Number> &x) const
{
  AssertDimension(y.size(), x.size());
  AssertDimension(x.size(), D1.get_n());

  D1.vmult(y, x);
  D2.vmult_add(y, x);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_HMATRIX_HELMHOLTZ_ACOUSTIC_HYPERSINGULAR_H_