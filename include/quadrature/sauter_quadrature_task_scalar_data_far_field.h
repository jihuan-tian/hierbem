// Copyright (C) 2006 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file sauter_quadrature_task_scalar_data_far_field.h
 * @brief Class for the scalar data used in Sauter quadrature tasks when
 * assembling far field H-matrix blocks.
 * @ingroup sauter_quadrature
 *
 * @date 2026-06-23
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_SCALAR_DATA_FAR_FIELD_H_
#define HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_SCALAR_DATA_FAR_FIELD_H_

#include <deal.II/base/types.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * This class is used as a wrapper for small scalar values that are
 * used in each quadrature task for far field H-matrix blocks.
 */
class SauterQuadratureTaskScalarDataFarField
{
public:
  /**
   * Local cell index of \f$K_x\f$, which is used when the H-matrix is built on
   * a subdomain. It is used to access precomputed cell dependent data at
   * quadrature points.
   */
  types::global_cell_index kx_cell_index_local;
  /**
   * Local cell index of \f$K_y\f$, which is used when the H-matrix is built on
   * a subdomain. It is used to access precomputed cell dependent data at
   * quadrature points.
   */
  types::global_cell_index ky_cell_index_local;
  /**
   * Cell local DoF index in the cell \f$K_x\f$.
   *
   * Given a full DoF index, find its index in the permuted list of DoF
   * indices in the cell \f$K_x\f$.
   */
  unsigned int i_local_index;
  /**
   * Cell local DoF index in the cell \f$K_y\f$.
   *
   * Given a full DoF index, find its index in the permuted list of DoF
   * indices in the cell \f$K_y\f$.
   */
  unsigned int j_local_index;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_SCALAR_DATA_FAR_FILED_H_
