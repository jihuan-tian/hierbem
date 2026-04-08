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
 * @file sauter_quadrature_task_scalar_data.h
 * @brief Class for the scalar data used in Sauter quadrature tasks.
 * @ingroup sauter_quadrature
 *
 * @date 2026-03-14
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_SCALAR_DATA_H_
#define HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_SCALAR_DATA_H_

#include <deal.II/base/types.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * This class is used as a wrapper for small scalar values that are
 * used in each quadrature task.
 */
class SauterQuadratureTaskScalarData
{
public:
  /**
   * Cell index of \f$K_x\f$.
   */
  types::global_cell_index kx_cell_index;
  /**
   * Cell index of \f$K_y\f$.
   */
  types::global_cell_index ky_cell_index;
  /**
   * Given a full DoF index, find its index in the permuted list of DoF
   * indices in the cell \f$K_x\f$.
   */
  unsigned int i_local_index;
  /**
   * Given a full DoF index, find its index in the permuted list of DoF
   * indices in the cell \f$K_y\f$.
   */
  unsigned int j_local_index;
  /**
   * Index of the @p MappingInfo object for the cell \f$K_x\f$.
   */
  unsigned int kx_mapping_index;
  /**
   * Index of the @p MappingInfo object for the cell \f$K_y\f$.
   */
  unsigned int ky_mapping_index;
  /**
   * Number of shape functions in the mapping for \f$K_x\f$.
   */
  unsigned int kx_mapping_n_shape_functions;
  /**
   * Number of shape functions in the mapping for \f$K_y\f$.
   */
  unsigned int ky_mapping_n_shape_functions;
  /**
   * Whether the normal vector of \f$K_x\f$ points into the volume, which
   * should be negated.
   */
  bool is_kx_normal_inward;
  /**
   * Whether the normal vector of \f$K_y\f$ points into the volume, which
   * should be negated.
   */
  bool is_ky_normal_inward;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_SCALAR_DATA_H_
