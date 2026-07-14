// Copyright (C) 2022-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file bem_values.h
 * @brief
 *
 * @date 2022-02-23
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_BEM_BEM_VALUES_H_
#define HIERBEM_INCLUDE_BEM_BEM_VALUES_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria.h>

#include <cuda_runtime.h>
#include <tbb/tbb.h>

#include "bem_tools.h"
#include "config.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "mapping/mapping_info.h"
#include "platform_shared/bem_tools.h"
#include "platform_shared/utilities.h"
#include "quadrature/sauter_quadrature_tools.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Values related to a pair of cells (panels) used in Galerkin BEM, which can
 * be considered as a counterpart of the @p FEValues for FEM in deal.ii.
 *
 * \mynote{The values encapsulated in @p BEMValues are shape functions value
 * and their gradient values at each quadrature point, as well as Sauter
 * quadrature rules for different cell neighboring types. These values will
 * be precalculated for improving the performance.}
 */
template <int dim, int spacedim, typename RangeNumberType = double>
class BEMValues
{
public:
  using FE_Poly_short = FE_Poly<dim, spacedim>;

  /**
   * Constructor
   *
   * \mynote{N.B. There is no default constructor for the class
   * @p BEMValues, because all the internal references to finite
   * element objects and quadrature objects should be initialized once the
   * @p BEMValues object is declared.}
   *
   * @param kx_fe
   * @param ky_fe
   * @param mappings A list of pointers to @p MappingInfo objects from the 1st
   * order to the highest order mapping.
   * @param mapping_support_point_table A table of mapping support points for
   * all active cells in the triangulation.
   * @param quad_rule_for_same_panel 4D Gauss quadrature object
   * @param quad_rule_for_common_edge 4D Gauss quadrature object
   * @param quad_rule_for_common_vertex 4D Gauss quadrature object
   * @param quad_rule_for_regular 2D Gauss quadrature object. N.B. 4D is not
   * used because there is no transformation from the Sauter parametric space to
   * the product space \f$\hat{K}_x \times \hat{K}_y\f$ of two unit cells.
   */
  BEMValues(const FiniteElement<dim, spacedim>              &kx_fe,
            const FiniteElement<dim, spacedim>              &ky_fe,
            const std::vector<MappingInfo<dim, spacedim> *> &mappings,
            const Table<2, Point<spacedim, RangeNumberType>>
                                  &mapping_support_point_table,
            const QGauss<dim * 2> &quad_rule_for_same_panel,
            const QGauss<dim * 2> &quad_rule_for_common_edge,
            const QGauss<dim * 2> &quad_rule_for_common_vertex,
            const QGauss<dim>     &quad_rule_for_regular,
            const bool             is_surface_curl_needed);


  /**
   * Copy constructor is deleted, since some members in this class, such as
   * finite elements, quadratures, mappings are references to external
   * objects.
   */
  BEMValues(const BEMValues<dim, spacedim, RangeNumberType> &bem_values) =
    delete;


  /**
   * Calculate the table storing shape function values and derivatives for
   * both finite element and mapping objects at Sauter quadrature points for
   * the same panel case.
   */
  void
  shape_function_values_same_panel();


  /**
   * Calculate the table storing shape function values and derivatives for
   * both finite element and mapping objects at Sauter quadrature points for
   * the common edge case.
   */
  void
  shape_function_values_common_edge();


  /**
   * Calculate the table storing shape function values and derivatives for
   * both finite element and mapping objects at Sauter quadrature points for
   * the common vertex case.
   */
  void
  shape_function_values_common_vertex();


  /**
   * Calculate the table storing shape function values and derivatives for
   * both finite element and mapping objects at Sauter quadrature points for
   * the regular case.
   */
  void
  shape_function_values_regular();

#pragma region ****Finite elements for the test space and trial space ****
  /**
   * Reference to finite element on the field cell \f$K_x\f$.
   */
  const FiniteElement<dim, spacedim> &kx_fe;
  /**
   * Reference to finite element on the field cell \f$K_y\f$.
   */
  const FiniteElement<dim, spacedim> &ky_fe;
#pragma endregion

#pragma region ****References to mapping objects and mapping support points ****
  /**
   * Mappings of from smallest to maximum order.
   */
  const std::vector<MappingInfo<dim, spacedim> *> &mappings;
  /**
   * Mapping support points for all active cells on the highest level in the
   * triangulation.
   *
   * Dim1: cell index
   * Dim2: mapping support point index, which is in the lexicographic order. The
   * size of this dimension is allocated to the number of mapping support points
   * in the mapping object with the highest order.
   */
  const Table<2, Point<spacedim, RangeNumberType>> &mapping_support_point_table;
#pragma endregion

#pragma region ****Quadrature rules for various cell neighboring types ****
  /**
   * Reference to 4D Sauter quadrature rule for the case that \f$K_x \equiv
   * K_y\f$.
   */
  const QGauss<dim * 2> &quad_rule_for_same_panel;
  /**
   * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and
   * \f$K_y\f$ share a common edge.
   */
  const QGauss<dim * 2> &quad_rule_for_common_edge;
  /**
   * Reference to 4D Sauter quadrature rule for the case that \f$K_x\f$ and
   * \f$K_y\f$ share a common vertex.
   */
  const QGauss<dim * 2> &quad_rule_for_common_vertex;
  /**
   * Reference to 2D Sauter quadrature rule for the case that \f$K_x\f$ and
   * \f$K_y\f$ are separated.
   */
  const QGauss<dim> &quad_rule_for_regular;
#pragma endregion

#pragma region ****Data tables for finite element shape functions evaluated at \
  quadrature points in the unit cell ****
  /**
   * Data table of finite element shape function values for \f$K_x\f$ in the
   * same panel case. It has three dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. \f$k_3\f$ term index: size=8
   * 3. Quadrature point index: size=number of quadrature points
   */
  Table<3, RangeNumberType> kx_shape_value_table_for_same_panel;
  /**
   * Data table of finite element shape function values for \f$K_y\f$ in the
   * same panel case. It has three dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. \f$k_3\f$ term index: size=8
   * 3. Quadrature point index: size=number of quadrature points
   */
  Table<3, RangeNumberType> ky_shape_value_table_for_same_panel;
  /**
   * Data table of finite element shape function values for \f$K_x\f$ in the
   * common edge case. It has three dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. \f$k_3\f$ term index: size=6
   * 3. Quadrature point index: size=number of quadrature points
   */
  Table<3, RangeNumberType> kx_shape_value_table_for_common_edge;
  /**
   * Data table of finite element shape function values for \f$K_y\f$ in the
   * common edge case. It has three dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. \f$k_3\f$ term index: size=6
   * 3. Quadrature point index: size=number of quadrature points
   */
  Table<3, RangeNumberType> ky_shape_value_table_for_common_edge;
  /**
   * Data table of finite element shape function values for \f$K_x\f$ in the
   * common vertex case. It has three dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. \f$k_3\f$ term index: size=4
   * 3. Quadrature point index: size=number of quadrature points
   */
  Table<3, RangeNumberType> kx_shape_value_table_for_common_vertex;
  /**
   * Data table of finite element shape function values for \f$K_y\f$ in the
   * common vertex case. It has three dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. \f$k_3\f$ term index: size=4
   * 3. Quadrature point index: size=number of quadrature points
   */
  Table<3, RangeNumberType> ky_shape_value_table_for_common_vertex;
  /**
   * Data table of finite element shape function values for \f$K_x\f$ in the
   * regular case. It has two dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. Quadrature point index: size=number of quadrature points
   */
  Table<2, RangeNumberType> kx_shape_value_table_for_regular;
  /**
   * Data table of finite element shape function values for \f$K_y\f$ in the
   * regular case. It has two dimensions:
   * 1. shape function index: size=@p dofs_per_cell
   * 2. Quadrature point index: size=number of quadrature points
   */
  Table<2, RangeNumberType> ky_shape_value_table_for_regular;
#pragma endregion

#pragma region ****Data tables for mapping shape functions evaluated at \
  quadrature points in the unit cell ****
  /**
   * Data table of mapping shape function values for \f$K_x\f$ in the same
   * panel case. It has 4 dimensions:
   * 1. \f$k_3\f$ term index: size=8
   * 2. Quadrature point index: size=number of quadrature points
   * 3. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 4. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<4, RangeNumberType> kx_mapping_shape_value_table_for_same_panel;

  /**
   * Data table of mapping shape function values for \f$K_y\f$ in the same
   * panel case. It has 4 dimensions:
   * 1. \f$k_3\f$ term index: size=8
   * 2. Quadrature point index: size=number of quadrature points
   * 3. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 4. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<4, RangeNumberType> ky_mapping_shape_value_table_for_same_panel;

  /**
   * Data table of mapping shape function values for \f$K_x\f$ in the common
   * edge case. It has 4 dimensions:
   * 1. \f$k_3\f$ term index: size=6
   * 2. Quadrature point index: size=number of quadrature points
   * 3. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 4. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<4, RangeNumberType> kx_mapping_shape_value_table_for_common_edge;

  /**
   * Data table of mapping shape function values for \f$K_y\f$ in the common
   * edge case. It has 4 dimensions:
   * 1. \f$k_3\f$ term index: size=6
   * 2. Quadrature point index: size=number of quadrature points
   * 3. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 4. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<4, RangeNumberType> ky_mapping_shape_value_table_for_common_edge;

  /**
   * Data table of mapping shape function values for \f$K_x\f$ in the common
   * vertex case. It has 4 dimensions:
   * 1. \f$k_3\f$ term index: size=4
   * 2. Quadrature point index: size=number of quadrature points
   * 3. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 4. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<4, RangeNumberType> kx_mapping_shape_value_table_for_common_vertex;

  /**
   * Data table of mapping shape function values for \f$K_y\f$ in the common
   * vertex case. It has 4 dimensions:
   * 1. \f$k_3\f$ term index: size=4
   * 2. Quadrature point index: size=number of quadrature points
   * 3. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 4. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<4, RangeNumberType> ky_mapping_shape_value_table_for_common_vertex;

  /**
   * Data table of mapping shape function values at 2D unit quadrature points in
   * the regular case.
   *
   * Because there is no coordinate transformation from the Sauter parametric
   * space to the product unit cell space \f$\hat{K}_x \times \hat{K}_y\f$,
   * there are no differences between values in \f$K_x\f$ and \f$K_y\f$ anymore.
   * Meanwhile, this data table is evaluated at each quadrature point in
   * <tt>QGauss<dim></tt>, which is different from <tt>QGauss<dim*2></tt> used
   * by other data tables.
   *
   * It has 3 dimensions:
   * 1. Quadrature point index: size=number of quadrature points
   * 2. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 3. Shape function index: size=number of shape functions in the highest
   * order mapping object
   */
  Table<3, RangeNumberType> mapping_shape_value_table_for_regular;
#pragma endregion

#pragma region ****Data tables for the gradient of finite element shape \
  functions evaluated at quadrature points in the unit cell ****
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_x\f$ in the same panel case. It has two dimensions:
   * 1. \f$k_3\f$ term index: size=8
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$dofs_per_cell*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    kx_shape_grad_matrix_table_for_same_panel;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_y\f$ in the same panel case. It has two dimensions:
   * 1. \f$k_3\f$ term index: size=8
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$dofs_per_cell*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    ky_shape_grad_matrix_table_for_same_panel;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_x\f$ in the common edge case. It has two dimensions:
   * 1. \f$k_3\f$ term index: size=6
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$dofs_per_cell*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    kx_shape_grad_matrix_table_for_common_edge;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_y\f$ in the common edge case. It has two dimensions:
   * 1. \f$k_3\f$ term index: size=6
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$dofs_per_cell*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    ky_shape_grad_matrix_table_for_common_edge;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_x\f$ in the common vertex case. It has two dimensions:
   * 1. \f$k_3\f$ term index: size=4
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$dofs_per_cell*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    kx_shape_grad_matrix_table_for_common_vertex;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_y\f$ in the common vertex case. It has two dimensions:
   * 1. \f$k_3\f$ term index: size=4
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$dofs_per_cell*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    ky_shape_grad_matrix_table_for_common_vertex;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_x\f$ in the regular case. It has one dimension: Quadrature point
   * index: size=number of quadrature points N.B. Each data item in the table is
   * itself a matrix with the dimension \f$dofs_per_cell*dim\f$.
   */
  Table<1, LAPACKFullMatrixExt<RangeNumberType>>
    kx_shape_grad_matrix_table_for_regular;
  /**
   * Data table of finite element shape function's gradient values for
   * \f$K_y\f$ in the regular case. It has one dimension: Quadrature point
   * index: size=number of quadrature points. N.B. Each data item in the table
   * is itself a matrix with the dimension \f$dofs_per_cell*dim\f$.
   */
  Table<1, LAPACKFullMatrixExt<RangeNumberType>>
    ky_shape_grad_matrix_table_for_regular;
#pragma endregion

#pragma region ****Data tables for the gradient of mapping shape functions \
  evaluated at quadrature points in the unit cell ****
  /**
   * Data table of mapping shape function's gradient values for
   * \f$K_x\f$ in the same panel case. It has 3 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. \f$k_3\f$ term index: size=8
   * 3. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<3, LAPACKFullMatrixExt<RangeNumberType>>
    kx_mapping_shape_grad_matrix_table_for_same_panel;

  /**
   * Data table of mapping shape function's gradient values for
   * \f$K_y\f$ in the same panel case. It has 3 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. \f$k_3\f$ term index: size=8
   * 3. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<3, LAPACKFullMatrixExt<RangeNumberType>>
    ky_mapping_shape_grad_matrix_table_for_same_panel;

  /**
   * Data table of mapping shape function's gradient values for
   * \f$K_x\f$ in the common edge case. It has 3 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. \f$k_3\f$ term index: size=6
   * 3. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<3, LAPACKFullMatrixExt<RangeNumberType>>
    kx_mapping_shape_grad_matrix_table_for_common_edge;

  /**
   * Data table of mapping shape function's gradient values for
   * \f$K_y\f$ in the common edge case. It has 3 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. \f$k_3\f$ term index: size=6
   * 3. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<3, LAPACKFullMatrixExt<RangeNumberType>>
    ky_mapping_shape_grad_matrix_table_for_common_edge;

  /**
   * Data table of mapping shape function's gradient values for
   * \f$K_x\f$ in the common vertex case. It has 3 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. \f$k_3\f$ term index: size=4
   * 3. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<3, LAPACKFullMatrixExt<RangeNumberType>>
    kx_mapping_shape_grad_matrix_table_for_common_vertex;

  /**
   * Data table of mapping shape function's gradient values for
   * \f$K_y\f$ in the common vertex case. It has 3 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. \f$k_3\f$ term index: size=4
   * 3. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<3, LAPACKFullMatrixExt<RangeNumberType>>
    ky_mapping_shape_grad_matrix_table_for_common_vertex;

  /**
   * Data table of mapping shape function's gradient values at 2D unit
   * quadrature points in the regular case.
   *
   * Because there is no coordinate transformation from the Sauter parametric
   * space to the product unit cell space \f$\hat{K}_x \times \hat{K}_y\f$,
   * there are no differences between values in \f$K_x\f$ and \f$K_y\f$ anymore.
   * Meanwhile, this data table is evaluated at each quadrature points in
   * <tt>QGauss<dim></tt>, which is different from <tt>QGauss<dim*2></tt> used
   * by other data tables.
   *
   * It has 2 dimensions:
   * 1. Mapping index, which is mapping order minus 1. It is used to get the
   * corresponding pointer to the MappingInfo object in BEMValues::mappings.
   * 2. Quadrature point index: size=number of quadrature points
   * N.B. Each data item in the table is itself a matrix with the dimension
   * \f$MappingQ::InternalData.n_shape_functions*dim\f$.
   */
  Table<2, LAPACKFullMatrixExt<RangeNumberType>>
    mapping_shape_grad_matrix_table_for_regular;
#pragma endregion

#pragma region ****Precomputed data for all cells in the triangulation, \
  used for the regular cell neighboring type ****
  /**
   * Jacobian values scaled by quadrature weights at quadrature points in used
   * cells.
   *
   * This data table is only used for the regular cell neighboring type.
   *
   * It has two dimensions:
   * 1. Local cell index.
   * 2. Quadrature point index.
   */
  Table<2, RangeNumberType> JxW_at_quad_points_for_regular;
  /**
   * Normal vectors at quadrature points in used cells.
   *
   * This data table is only used for the regular cell neighboring type.
   *
   * It has two dimensions:
   * 1. Local cell index.
   * 2. Quadrature point index.
   */
  Table<2, Tensor<1, spacedim, RangeNumberType>>
    normals_at_quad_points_for_regular;
  /**
   * Quadrature point coordinates in used cells.
   *
   * This data table is only used for the regular cell neighboring type.
   *
   * It has two dimensions:
   * 1. Local cell index.
   * 2. Quadrature point index.
   */
  Table<2, Point<spacedim, RangeNumberType>> quad_points_for_regular;
  /**
   * Whether surface curls of finite element shape functions should be computed.
   */
  bool is_surface_curl_needed;
  /**
   * Surface gradient at quadrature points in used cells for the test space.
   *
   * This data table is only used for the regular cell neighboring type.
   *
   * It has three dimensions:
   * 1. Local cell index.
   * 2. Shape function index of the finite element.
   * 3. Quadrature point index.
   */
  Table<3, Tensor<1, spacedim, RangeNumberType>>
    kx_shape_curls_at_quad_points_for_regular;
  /**
   * Surface gradient at quadrature points in used cells for the trial space.
   *
   * This data table is only used for the regular cell neighboring type.
   *
   * It has three dimensions:
   * 1. Local cell index.
   * 2. Shape function index of the finite element.
   * 3. Quadrature point index.
   */
  Table<3, Tensor<1, spacedim, RangeNumberType>>
    ky_shape_curls_at_quad_points_for_regular;
#pragma endregion

  /**
   * Fill the data tables for the values and derivatives of finite element
   * shape functions and mapping object shape functions.
   */
  void
  fill_shape_function_value_tables();

  /**
   * Compute cell values at quadrature points for all active cells in a
   * triangulation, which will be used in the regular cell neighboring type.
   *
   * This function is only used for building BEM full matrices on the whole
   * triangulation, which does not use CUDA. And it should be called after
   * calling
   * @p fill_shape_function_value_tables or @p shape_function_values_regular.
   *
   * @param tria Triangulation
   * @param mapping_indices A list of mapping indices for all active cells in
   * the triangulation
   * @param normal_detector An object with a member function named
   * @p is_normal_vector_inward, which is used to determine if the normal vector
   * of a cell points into (true) or outside from a associated volume.
   */
  template <typename SurfaceNormalDetector>
  void
  compute_bilinear_form_cell_values_for_regular(
    const Triangulation<dim, spacedim> &tria,
    const std::vector<unsigned int>    &mapping_indices,
    const SurfaceNormalDetector        &normal_detector);

  /**
   * Compute cell values at quadrature points for all cells used by a bilinear
   * form, which will be used in the regular cell neighboring type.
   *
   * This function is only used for building BEM full matrices on a subdomain,
   * which does not use CUDA. And it should be called after calling
   * @p fill_shape_function_value_tables or @p shape_function_values_regular.
   *
   * @param cell_iterator_ptrs A list of pointers to cell iterators used by a
   * bilinear form.
   * @param local_to_global_cell_index_map Vector used as a map from local
   * cell indices to global cell indices.
   * @param mapping_indices A list of mapping indices for all active cells in
   * the triangulation
   * @param normal_detector An object with a member function named
   * @p is_normal_vector_inward, which is used to determine if the normal vector
   * of a cell points into (true) or outside from a associated volume.
   */
  template <typename SurfaceNormalDetector>
  void
  compute_bilinear_form_cell_values_for_regular(
    const std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>
                                                &cell_iterator_ptrs,
    const std::vector<types::global_cell_index> &local_to_global_cell_index_map,
    const std::vector<unsigned int>             &mapping_indices,
    const SurfaceNormalDetector                 &normal_detector);

protected:
  /**
   * Initialize the data tables for the finite element shape function values.
   */
  void
  init_shape_value_tables();

  /**
   * Initialize the data tables for the mapping shape function values.
   *
   * \mynote{The number of mapping shape function in the highest order mapping
   * object is used to initialize the second dimension of each data table.}
   */
  void
  init_mapping_shape_value_tables();

  /**
   * Initialize the data tables for the gradient values of finite element
   * shape functions.
   */
  void
  init_shape_grad_matrix_tables();

  /**
   * Initialize the data tables for the gradient values of shape functions in
   * the mapping object.
   */
  void
  init_mapping_shape_grad_matrix_tables();

  /**
   * Initialize matrices storing the gradient values of shape functions in the
   * mapping object.
   *
   * This version is used for the same panel, common edge and common vertex
   * cell neighboring types.
   *
   * @param table
   */
  void
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    Table<3, LAPACKFullMatrixExt<RangeNumberType>> &table);

  /**
   * Initialize matrices storing the gradient values of shape functions in the
   * mapping object.
   *
   * This version is used for the regular cell neighboring type.
   *
   * @param table
   */
  void
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> &table);

  /**
   * Resize the internal data in all orders of mappings.
   *
   * \mynote{The InternalData within a MappingQ object is a temporary place
   * holding the shape function values and their derivatives. Because deal.ii
   * is an FEM library, which does not handle different \f$k_3\f$ terms and
   * cell neighboring types as in the Sauter quadrature used in BEM, whenever
   * we come to a kind of cell neighboring type which has its associated
   * quadrature object, we need to reinitialize the InternalData objects in
   * all possible orders of mappings and then compute shape function values
   * and their derivatives. After the computation, the values within
   * InternalData will be copied into related data tables in BEMValues.}
   *
   * @pre
   * @post
   * @param n_q_points
   */
  void
  resize_internal_data_in_mappings(const unsigned int n_q_points) const;

  /**
   * Compute cell values for all finite element shape functions at a single
   * quadrature point, which will be used in the regular cell neighboring type.
   */
  void
  compute_cell_values_at_a_quad_point_for_regular(
    const unsigned int             quad_no,
    const types::global_cell_index cell_index_local,
    const types::global_cell_index cell_index_global,
    const bool                     cell_normals_inward_flag,
    const unsigned int             mapping_index,
    const unsigned int             mapping_n_shape_functions);
};


template <int dim, int spacedim, typename RangeNumberType>
BEMValues<dim, spacedim, RangeNumberType>::BEMValues(
  const FiniteElement<dim, spacedim>               &kx_fe,
  const FiniteElement<dim, spacedim>               &ky_fe,
  const std::vector<MappingInfo<dim, spacedim> *>  &mappings,
  const Table<2, Point<spacedim, RangeNumberType>> &mapping_support_point_table,
  const QGauss<dim * 2>                            &quad_rule_for_same_panel,
  const QGauss<dim * 2>                            &quad_rule_for_common_edge,
  const QGauss<dim * 2>                            &quad_rule_for_common_vertex,
  const QGauss<dim>                                &quad_rule_for_regular,
  const bool                                        is_surface_curl_needed)
  : kx_fe(kx_fe)
  , ky_fe(ky_fe)
  , mappings(mappings)
  , mapping_support_point_table(mapping_support_point_table)
  , quad_rule_for_same_panel(quad_rule_for_same_panel)
  , quad_rule_for_common_edge(quad_rule_for_common_edge)
  , quad_rule_for_common_vertex(quad_rule_for_common_vertex)
  , quad_rule_for_regular(quad_rule_for_regular)
  , is_surface_curl_needed(is_surface_curl_needed)
{
  init_shape_value_tables();
  init_mapping_shape_value_tables();
  if (is_surface_curl_needed)
    init_shape_grad_matrix_tables();
  init_mapping_shape_grad_matrix_tables();
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::init_shape_value_tables()
{
  kx_shape_value_table_for_same_panel.reinit(
    TableIndices<3>(kx_fe.dofs_per_cell, 8, quad_rule_for_same_panel.size()));
  ky_shape_value_table_for_same_panel.reinit(
    TableIndices<3>(ky_fe.dofs_per_cell, 8, quad_rule_for_same_panel.size()));

  kx_shape_value_table_for_common_edge.reinit(
    TableIndices<3>(kx_fe.dofs_per_cell, 6, quad_rule_for_common_edge.size()));
  ky_shape_value_table_for_common_edge.reinit(
    TableIndices<3>(ky_fe.dofs_per_cell, 6, quad_rule_for_common_edge.size()));

  kx_shape_value_table_for_common_vertex.reinit(TableIndices<3>(
    kx_fe.dofs_per_cell, 4, quad_rule_for_common_vertex.size()));
  ky_shape_value_table_for_common_vertex.reinit(TableIndices<3>(
    ky_fe.dofs_per_cell, 4, quad_rule_for_common_vertex.size()));

  kx_shape_value_table_for_regular.reinit(
    TableIndices<2>(kx_fe.dofs_per_cell, quad_rule_for_regular.size()));
  ky_shape_value_table_for_regular.reinit(
    TableIndices<2>(ky_fe.dofs_per_cell, quad_rule_for_regular.size()));
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::init_mapping_shape_value_tables()
{
  const unsigned int max_mapping_n_shape_functions =
    mappings.back()->get_data()->n_shape_functions;

  kx_mapping_shape_value_table_for_same_panel.reinit(
    TableIndices<4>(8,
                    quad_rule_for_same_panel.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));

  kx_mapping_shape_value_table_for_common_edge.reinit(
    TableIndices<4>(6,
                    quad_rule_for_common_edge.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));

  kx_mapping_shape_value_table_for_common_vertex.reinit(
    TableIndices<4>(4,
                    quad_rule_for_common_vertex.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));

  ky_mapping_shape_value_table_for_same_panel.reinit(
    TableIndices<4>(8,
                    quad_rule_for_same_panel.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));

  ky_mapping_shape_value_table_for_common_edge.reinit(
    TableIndices<4>(6,
                    quad_rule_for_common_edge.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));

  ky_mapping_shape_value_table_for_common_vertex.reinit(
    TableIndices<4>(4,
                    quad_rule_for_common_vertex.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));

  mapping_shape_value_table_for_regular.reinit(
    TableIndices<3>(quad_rule_for_regular.size(),
                    mappings.size(),
                    max_mapping_n_shape_functions));
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::init_shape_grad_matrix_tables()
{
  kx_shape_grad_matrix_table_for_same_panel.reinit(
    TableIndices<2>(8, quad_rule_for_same_panel.size()));
  ky_shape_grad_matrix_table_for_same_panel.reinit(
    TableIndices<2>(8, quad_rule_for_same_panel.size()));

  kx_shape_grad_matrix_table_for_common_edge.reinit(
    TableIndices<2>(6, quad_rule_for_common_edge.size()));
  ky_shape_grad_matrix_table_for_common_edge.reinit(
    TableIndices<2>(6, quad_rule_for_common_edge.size()));

  kx_shape_grad_matrix_table_for_common_vertex.reinit(
    TableIndices<2>(4, quad_rule_for_common_vertex.size()));
  ky_shape_grad_matrix_table_for_common_vertex.reinit(
    TableIndices<2>(4, quad_rule_for_common_vertex.size()));

  kx_shape_grad_matrix_table_for_regular.reinit(
    TableIndices<1>(quad_rule_for_regular.size()));
  ky_shape_grad_matrix_table_for_regular.reinit(
    TableIndices<1>(quad_rule_for_regular.size()));
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::
  init_mapping_shape_grad_matrix_tables()
{
  kx_mapping_shape_grad_matrix_table_for_same_panel.reinit(
    TableIndices<3>(mappings.size(), 8, quad_rule_for_same_panel.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    kx_mapping_shape_grad_matrix_table_for_same_panel);

  kx_mapping_shape_grad_matrix_table_for_common_edge.reinit(
    TableIndices<3>(mappings.size(), 6, quad_rule_for_common_edge.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    kx_mapping_shape_grad_matrix_table_for_common_edge);

  kx_mapping_shape_grad_matrix_table_for_common_vertex.reinit(
    TableIndices<3>(mappings.size(), 4, quad_rule_for_common_vertex.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    kx_mapping_shape_grad_matrix_table_for_common_vertex);

  ky_mapping_shape_grad_matrix_table_for_same_panel.reinit(
    TableIndices<3>(mappings.size(), 8, quad_rule_for_same_panel.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    ky_mapping_shape_grad_matrix_table_for_same_panel);

  ky_mapping_shape_grad_matrix_table_for_common_edge.reinit(
    TableIndices<3>(mappings.size(), 6, quad_rule_for_common_edge.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    ky_mapping_shape_grad_matrix_table_for_common_edge);

  ky_mapping_shape_grad_matrix_table_for_common_vertex.reinit(
    TableIndices<3>(mappings.size(), 4, quad_rule_for_common_vertex.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    ky_mapping_shape_grad_matrix_table_for_common_vertex);

  mapping_shape_grad_matrix_table_for_regular.reinit(
    TableIndices<2>(mappings.size(), quad_rule_for_regular.size()));
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    mapping_shape_grad_matrix_table_for_regular);
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    Table<3, LAPACKFullMatrixExt<RangeNumberType>> &table)
{
  for (unsigned int m = 0; m < table.size(0); m++)
    for (unsigned int i = 0; i < table.size(1); i++)
      for (unsigned int j = 0; j < table.size(2); j++)
        table(m, i, j).reinit(mappings[m]->get_data()->n_shape_functions, dim);
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::
  init_internal_matrix_in_mapping_shape_grad_matrix_table(
    Table<2, LAPACKFullMatrixExt<RangeNumberType>> &table)
{
  for (unsigned int m = 0; m < table.size(0); m++)
    for (unsigned int i = 0; i < table.size(1); i++)
      table(m, i).reinit(mappings[m]->get_data()->n_shape_functions, dim);
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::resize_internal_data_in_mappings(
  const unsigned int n_q_points) const
{
  for (auto m : mappings)
    m->resize_internal_data(n_q_points);
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::fill_shape_function_value_tables()
{
  shape_function_values_same_panel();
  shape_function_values_common_edge();
  shape_function_values_common_vertex();
  shape_function_values_regular();
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_same_panel()
{
  const unsigned int                  kx_dofs_per_cell = kx_fe.dofs_per_cell;
  const unsigned int                  ky_dofs_per_cell = ky_fe.dofs_per_cell;
  [[maybe_unused]] const unsigned int max_mapping_n_shape_functions =
    mappings.back()->get_data()->n_shape_functions;
  const unsigned int n_q_points = quad_rule_for_same_panel.size();

  // Make assertion about the length for each dimension of the data table.
  AssertDimension(kx_shape_value_table_for_same_panel.size(0),
                  kx_dofs_per_cell);
  AssertDimension(kx_shape_value_table_for_same_panel.size(1), 8);
  AssertDimension(kx_shape_value_table_for_same_panel.size(2), n_q_points);

  AssertDimension(ky_shape_value_table_for_same_panel.size(0),
                  ky_dofs_per_cell);
  AssertDimension(ky_shape_value_table_for_same_panel.size(1), 8);
  AssertDimension(ky_shape_value_table_for_same_panel.size(2), n_q_points);

  AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(0), 8);
  AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(1),
                  n_q_points);
  AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(2),
                  mappings.size());
  AssertDimension(kx_mapping_shape_value_table_for_same_panel.size(3),
                  max_mapping_n_shape_functions);

  AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(0), 8);
  AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(1),
                  n_q_points);
  AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(2),
                  mappings.size());
  AssertDimension(ky_mapping_shape_value_table_for_same_panel.size(3),
                  max_mapping_n_shape_functions);

  if (is_surface_curl_needed)
    {
      AssertDimension(kx_shape_grad_matrix_table_for_same_panel.size(0), 8);
      AssertDimension(kx_shape_grad_matrix_table_for_same_panel.size(1),
                      n_q_points);

      AssertDimension(ky_shape_grad_matrix_table_for_same_panel.size(0), 8);
      AssertDimension(ky_shape_grad_matrix_table_for_same_panel.size(1),
                      n_q_points);
    }

  AssertDimension(kx_mapping_shape_grad_matrix_table_for_same_panel.size(0),
                  mappings.size());
  AssertDimension(kx_mapping_shape_grad_matrix_table_for_same_panel.size(1), 8);
  AssertDimension(kx_mapping_shape_grad_matrix_table_for_same_panel.size(2),
                  n_q_points);

  AssertDimension(ky_mapping_shape_grad_matrix_table_for_same_panel.size(0),
                  mappings.size());
  AssertDimension(ky_mapping_shape_grad_matrix_table_for_same_panel.size(1), 8);
  AssertDimension(ky_mapping_shape_grad_matrix_table_for_same_panel.size(2),
                  n_q_points);

  /**
   * Initialize the internal data in the mapping objects.
   */
  resize_internal_data_in_mappings(n_q_points);

  /**
   * Quadrature points in the Sauter's parametric space.
   */
  std::vector<Point<dim * 2>> quad_points =
    quad_rule_for_same_panel.get_points();

  /**
   * Get the polynomial space inverse numbering for accessing the shape
   * functions in the lexicographic order.
   *
   * \alert{Here I have adopted an assumption that the finite elements
   * are based on tensor product polynomials.}
   */
  const FE_Poly_short &kx_fe_poly = dynamic_cast<const FE_Poly_short &>(kx_fe);
  const FE_Poly_short &ky_fe_poly = dynamic_cast<const FE_Poly_short &>(ky_fe);

  std::vector<unsigned int> kx_poly_space_inverse_numbering(
    kx_fe_poly.get_poly_space_numbering_inverse());
  std::vector<unsigned int> ky_poly_space_inverse_numbering(
    ky_fe_poly.get_poly_space_numbering_inverse());

  /**
   * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
   * respectively.
   */
  std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
  std::vector<Point<dim>> ky_unit_quad_points(n_q_points);

  // Iterate over each $k_3$ part.
  for (unsigned k = 0; k < 8; k++)
    {
      // Iterate over each quadrature point.
      for (unsigned int q = 0; q < n_q_points; q++)
        {
          // Transform the quadrature point in the parametric space to
          // the unit cells of \f$K_x\f$ and \f$K_y\f$.
          sauter_same_panel_parametric_coords_to_unit_cells(
            quad_points[q], k, kx_unit_quad_points[q], ky_unit_quad_points[q]);

          /**
           * Iterate over each finite element shape function in the
           * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
           * at the current quadrature point @p kx_unit_quad_points[q].
           */
          for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
            {
              kx_shape_value_table_for_same_panel(TableIndices<3>(s, k, q)) =
                static_cast<RangeNumberType>(
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_unit_quad_points[q]));
            }

          /**
           * Iterate over each finite element shape function in the
           * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
           * at the current quadrature point @p ky_unit_quad_points[q].
           */
          for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
            {
              ky_shape_value_table_for_same_panel(TableIndices<3>(s, k, q)) =
                static_cast<RangeNumberType>(
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_unit_quad_points[q]));
            }

          if (is_surface_curl_needed)
            {
              // Calculate the Jacobian matrix evaluated at
              // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows
              // correspond to shape functions which are in the lexicographic
              // order.
              kx_shape_grad_matrix_table_for_same_panel(TableIndices<2>(k, q)) =
                BEMTools::shape_grad_matrix_in_lexicographic_order<
                  dim,
                  spacedim,
                  RangeNumberType>(kx_fe, kx_unit_quad_points[q]);
              // Calculate the Jacobian matrix evaluated at
              // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows
              // correspond to shape functions which are in the lexicographic
              // order.
              ky_shape_grad_matrix_table_for_same_panel(TableIndices<2>(k, q)) =
                BEMTools::shape_grad_matrix_in_lexicographic_order<
                  dim,
                  spacedim,
                  RangeNumberType>(ky_fe, ky_unit_quad_points[q]);
            }
        }

      /**
       * Compute mapping shape function values and their derivatives in batch.
       *
       * \alert{Even though the internally generated polynomials in the
       * mapping object are in the tensor product order, the shape function
       * values and derivatives within @p InternalData are still stored in the
       * hierarchic order. This can be verified by checking the source code of
       * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
       * (see
       * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
       *
       * However, this behavior is different from the documentation for the
       * function @p MappingQ< dim, spacedim >::InternalData::shape().}
       */
      for (unsigned int m = 0; m < mappings.size(); m++)
        {
          auto              &mapping_data = mappings[m]->get_data();
          const unsigned int mapping_n_shape_functions =
            mapping_data->n_shape_functions;
          // The numbering used for accessing shape function values and their
          // derivatives in the lexicographic order.
          const std::vector<unsigned int> &lexicographic_numbering =
            mappings[m]->get_lexicographic_numberings_for_shape_functions()[0];

          // Compute mapping shape function values and their derivatives for kx.
          mapping_data->compute_shape_function_values(kx_unit_quad_points);
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
                {
                  kx_mapping_shape_value_table_for_same_panel(
                    TableIndices<4>(k, q, m, s)) =
                    static_cast<RangeNumberType>(
                      mapping_data->shape(q, lexicographic_numbering[s]));

                  for (unsigned int d = 0; d < dim; d++)
                    {
                      kx_mapping_shape_grad_matrix_table_for_same_panel(
                        TableIndices<3>(m, k, q))(s, d) =
                        static_cast<RangeNumberType>(mapping_data->derivative(
                          q, lexicographic_numbering[s])[d]);
                    }
                }
            }

          // Compute mapping shape function values and their derivatives for ky.
          mapping_data->compute_shape_function_values(ky_unit_quad_points);
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
                {
                  ky_mapping_shape_value_table_for_same_panel(
                    TableIndices<4>(k, q, m, s)) =
                    static_cast<RangeNumberType>(
                      mapping_data->shape(q, lexicographic_numbering[s]));

                  for (unsigned int d = 0; d < dim; d++)
                    {
                      ky_mapping_shape_grad_matrix_table_for_same_panel(
                        TableIndices<3>(m, k, q))(s, d) =
                        static_cast<RangeNumberType>(mapping_data->derivative(
                          q, lexicographic_numbering[s])[d]);
                    }
                }
            }
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_common_edge()
{
  const unsigned int                  kx_dofs_per_cell = kx_fe.dofs_per_cell;
  const unsigned int                  ky_dofs_per_cell = ky_fe.dofs_per_cell;
  [[maybe_unused]] const unsigned int max_mapping_n_shape_functions =
    mappings.back()->get_data()->n_shape_functions;
  const unsigned int n_q_points = quad_rule_for_common_edge.size();

  // Make assertion about the length for each dimension of the data table.
  AssertDimension(kx_shape_value_table_for_common_edge.size(0),
                  kx_dofs_per_cell);
  AssertDimension(kx_shape_value_table_for_common_edge.size(1), 6);
  AssertDimension(kx_shape_value_table_for_common_edge.size(2), n_q_points);

  AssertDimension(ky_shape_value_table_for_common_edge.size(0),
                  ky_dofs_per_cell);
  AssertDimension(ky_shape_value_table_for_common_edge.size(1), 6);
  AssertDimension(ky_shape_value_table_for_common_edge.size(2), n_q_points);

  AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(0), 6);
  AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(1),
                  n_q_points);
  AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(2),
                  mappings.size());
  AssertDimension(kx_mapping_shape_value_table_for_common_edge.size(3),
                  max_mapping_n_shape_functions);

  AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(0), 6);
  AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(1),
                  n_q_points);
  AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(2),
                  mappings.size());
  AssertDimension(ky_mapping_shape_value_table_for_common_edge.size(3),
                  max_mapping_n_shape_functions);

  if (is_surface_curl_needed)
    {
      AssertDimension(kx_shape_grad_matrix_table_for_common_edge.size(0), 6);
      AssertDimension(kx_shape_grad_matrix_table_for_common_edge.size(1),
                      n_q_points);

      AssertDimension(ky_shape_grad_matrix_table_for_common_edge.size(0), 6);
      AssertDimension(ky_shape_grad_matrix_table_for_common_edge.size(1),
                      n_q_points);
    }

  AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_edge.size(0),
                  mappings.size());
  AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_edge.size(1),
                  6);
  AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_edge.size(2),
                  n_q_points);

  AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_edge.size(0),
                  mappings.size());
  AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_edge.size(1),
                  6);
  AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_edge.size(2),
                  n_q_points);

  /**
   * Initialize the internal data in the mapping objects.
   */
  resize_internal_data_in_mappings(n_q_points);

  /**
   * Quadrature points in the Sauter's parametric space.
   */
  std::vector<Point<dim * 2>> quad_points =
    quad_rule_for_common_edge.get_points();

  /**
   * Get the polynomial space inverse numbering for accessing the shape
   * functions in the lexicographic order.
   *
   * \alert{Here I have adopted an assumption that the finite elements are
   * based on tensor product polynomials.}
   */
  const FE_Poly_short &kx_fe_poly = dynamic_cast<const FE_Poly_short &>(kx_fe);
  const FE_Poly_short &ky_fe_poly = dynamic_cast<const FE_Poly_short &>(ky_fe);

  std::vector<unsigned int> kx_poly_space_inverse_numbering(
    kx_fe_poly.get_poly_space_numbering_inverse());
  std::vector<unsigned int> ky_poly_space_inverse_numbering(
    ky_fe_poly.get_poly_space_numbering_inverse());

  /**
   * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
   * respectively.
   */
  std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
  std::vector<Point<dim>> ky_unit_quad_points(n_q_points);

  // Iterate over each $k_3$ part.
  for (unsigned k = 0; k < 6; k++)
    {
      // Iterate over each quadrature point.
      for (unsigned int q = 0; q < n_q_points; q++)
        {
          // Transform the quadrature point in the parametric space to
          // the unit cells of \f$K_x\f$ and \f$K_y\f$.
          sauter_common_edge_parametric_coords_to_unit_cells(
            quad_points[q], k, kx_unit_quad_points[q], ky_unit_quad_points[q]);

          /**
           * Iterate over each finite element shape function in the
           * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
           * at the current quadrature point @p kx_unit_quad_points[q].
           */
          for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
            {
              kx_shape_value_table_for_common_edge(TableIndices<3>(s, k, q)) =
                static_cast<RangeNumberType>(
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_unit_quad_points[q]));
            }

          /**
           * Iterate over each finite element shape function in the
           * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
           * at the current quadrature point @p ky_unit_quad_points[q].
           */
          for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
            {
              ky_shape_value_table_for_common_edge(TableIndices<3>(s, k, q)) =
                static_cast<RangeNumberType>(
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_unit_quad_points[q]));
            }

          if (is_surface_curl_needed)
            {
              // Calculate the Jacobian matrix evaluated at
              // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows
              // correspond to shape functions which are in the lexicographic
              // order.
              kx_shape_grad_matrix_table_for_common_edge(
                TableIndices<2>(k, q)) =
                BEMTools::shape_grad_matrix_in_lexicographic_order<
                  dim,
                  spacedim,
                  RangeNumberType>(kx_fe, kx_unit_quad_points[q]);
              // Calculate the Jacobian matrix evaluated at
              // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows
              // correspond to shape functions which are in the lexicographic
              // order.
              ky_shape_grad_matrix_table_for_common_edge(
                TableIndices<2>(k, q)) =
                BEMTools::shape_grad_matrix_in_lexicographic_order<
                  dim,
                  spacedim,
                  RangeNumberType>(ky_fe, ky_unit_quad_points[q]);
            }
        }

      /**
       * Compute mapping shape function values and their derivatives in batch.
       *
       * \alert{Even though the internally generated polynomials in the
       * mapping object are in the tensor product order, the shape function
       * values and derivatives are still in the hierarchic order. This can be
       * verified by checking the source code of
       * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
       * (see
       * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
       *
       * However, this behavior is different from the documentation for the
       * function @p MappingQ< dim, spacedim >::InternalData::shape().}
       */
      for (unsigned int m = 0; m < mappings.size(); m++)
        {
          auto              &mapping_data = mappings[m]->get_data();
          const unsigned int mapping_n_shape_functions =
            mapping_data->n_shape_functions;
          // The numbering used for accessing shape function values and their
          // derivatives in the lexicographic order.
          const std::vector<unsigned int> &lexicographic_numbering =
            mappings[m]->get_lexicographic_numberings_for_shape_functions()[0];

          // Compute shape function values and their derivatives for kx.
          mapping_data->compute_shape_function_values(kx_unit_quad_points);
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
                {
                  kx_mapping_shape_value_table_for_common_edge(
                    TableIndices<4>(k, q, m, s)) =
                    static_cast<RangeNumberType>(
                      mapping_data->shape(q, lexicographic_numbering[s]));

                  for (unsigned int d = 0; d < dim; d++)
                    {
                      kx_mapping_shape_grad_matrix_table_for_common_edge(
                        TableIndices<3>(m, k, q))(s, d) =
                        static_cast<RangeNumberType>(mapping_data->derivative(
                          q, lexicographic_numbering[s])[d]);
                    }
                }
            }

          // Compute shape function values and their derivatives for ky.
          mapping_data->compute_shape_function_values(ky_unit_quad_points);
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
                {
                  ky_mapping_shape_value_table_for_common_edge(
                    TableIndices<4>(k, q, m, s)) =
                    static_cast<RangeNumberType>(
                      mapping_data->shape(q, lexicographic_numbering[s]));

                  for (unsigned int d = 0; d < dim; d++)
                    {
                      ky_mapping_shape_grad_matrix_table_for_common_edge(
                        TableIndices<3>(m, k, q))(s, d) =
                        static_cast<RangeNumberType>(mapping_data->derivative(
                          q, lexicographic_numbering[s])[d]);
                    }
                }
            }
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_common_vertex()
{
  const unsigned int                  kx_dofs_per_cell = kx_fe.dofs_per_cell;
  const unsigned int                  ky_dofs_per_cell = ky_fe.dofs_per_cell;
  [[maybe_unused]] const unsigned int max_mapping_n_shape_functions =
    mappings.back()->get_data()->n_shape_functions;
  const unsigned int n_q_points = quad_rule_for_common_vertex.size();

  // Make assertion about the length for each dimension of the data table.
  AssertDimension(kx_shape_value_table_for_common_vertex.size(0),
                  kx_dofs_per_cell);
  AssertDimension(kx_shape_value_table_for_common_vertex.size(1), 4);
  AssertDimension(kx_shape_value_table_for_common_vertex.size(2), n_q_points);

  AssertDimension(ky_shape_value_table_for_common_vertex.size(0),
                  ky_dofs_per_cell);
  AssertDimension(ky_shape_value_table_for_common_vertex.size(1), 4);
  AssertDimension(ky_shape_value_table_for_common_vertex.size(2), n_q_points);

  AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(0), 4);
  AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(1),
                  n_q_points);
  AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(2),
                  mappings.size());
  AssertDimension(kx_mapping_shape_value_table_for_common_vertex.size(3),
                  max_mapping_n_shape_functions);

  AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(0), 4);
  AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(1),
                  n_q_points);
  AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(2),
                  mappings.size());
  AssertDimension(ky_mapping_shape_value_table_for_common_vertex.size(3),
                  max_mapping_n_shape_functions);

  if (is_surface_curl_needed)
    {
      AssertDimension(kx_shape_grad_matrix_table_for_common_vertex.size(0), 4);
      AssertDimension(kx_shape_grad_matrix_table_for_common_vertex.size(1),
                      n_q_points);

      AssertDimension(ky_shape_grad_matrix_table_for_common_vertex.size(0), 4);
      AssertDimension(ky_shape_grad_matrix_table_for_common_vertex.size(1),
                      n_q_points);
    }

  AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_vertex.size(0),
                  mappings.size());
  AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_vertex.size(1),
                  4);
  AssertDimension(kx_mapping_shape_grad_matrix_table_for_common_vertex.size(2),
                  n_q_points);

  AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_vertex.size(0),
                  mappings.size());
  AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_vertex.size(1),
                  4);
  AssertDimension(ky_mapping_shape_grad_matrix_table_for_common_vertex.size(2),
                  n_q_points);

  /**
   * Initialize the internal data in the mapping objects.
   */
  resize_internal_data_in_mappings(n_q_points);

  /**
   * Quadrature points in the Sauter's parametric space.
   */
  std::vector<Point<dim * 2>> quad_points =
    quad_rule_for_common_vertex.get_points();

  /**
   * Get the polynomial space inverse numbering for accessing the shape
   * functions in the lexicographic order.
   *
   * \alert{Here I have adopted an assumption that the finite elements are
   * based on tensor product polynomials.}
   */
  const FE_Poly_short &kx_fe_poly = dynamic_cast<const FE_Poly_short &>(kx_fe);
  const FE_Poly_short &ky_fe_poly = dynamic_cast<const FE_Poly_short &>(ky_fe);

  std::vector<unsigned int> kx_poly_space_inverse_numbering(
    kx_fe_poly.get_poly_space_numbering_inverse());
  std::vector<unsigned int> ky_poly_space_inverse_numbering(
    ky_fe_poly.get_poly_space_numbering_inverse());

  /**
   * Quadrature points in the unit cells of \f$K_x\f$ and \f$K_y\f$
   * respectively.
   */
  std::vector<Point<dim>> kx_unit_quad_points(n_q_points);
  std::vector<Point<dim>> ky_unit_quad_points(n_q_points);

  // Iterate over each $k_3$ part.
  for (unsigned k = 0; k < 4; k++)
    {
      // Iterate over each quadrature point.
      for (unsigned int q = 0; q < n_q_points; q++)
        {
          // Transform the quadrature point in the parametric space to
          // the unit cells of \f$K_x\f$ and \f$K_y\f$.
          sauter_common_vertex_parametric_coords_to_unit_cells(
            quad_points[q], k, kx_unit_quad_points[q], ky_unit_quad_points[q]);

          /**
           * Iterate over each finite element shape function in the
           * lexicographic order on the unit cell of \f$K_x\f$ and evaluate it
           * at the current quadrature point @p kx_unit_quad_points[q].
           */
          for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
            {
              kx_shape_value_table_for_common_vertex(TableIndices<3>(s, k, q)) =
                static_cast<RangeNumberType>(
                  kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                                    kx_unit_quad_points[q]));
            }

          /**
           * Iterate over each finite element shape function in the
           * lexicographic order on the unit cell of \f$K_y\f$ and evaluate it
           * at the current quadrature point @p ky_unit_quad_points[q].
           */
          for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
            {
              ky_shape_value_table_for_common_vertex(TableIndices<3>(s, k, q)) =
                static_cast<RangeNumberType>(
                  ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                                    ky_unit_quad_points[q]));
            }

          if (is_surface_curl_needed)
            {
              // Calculate the Jacobian matrix evaluated at
              // <code>kx_quad_point</code> in  \f$K_x\f$. Matrix rows
              // correspond to shape functions which are in the lexicographic
              // order.
              kx_shape_grad_matrix_table_for_common_vertex(
                TableIndices<2>(k, q)) =
                BEMTools::shape_grad_matrix_in_lexicographic_order<
                  dim,
                  spacedim,
                  RangeNumberType>(kx_fe, kx_unit_quad_points[q]);
              // Calculate the Jacobian matrix evaluated at
              // <code>ky_quad_point</code> in  \f$K_y\f$. Matrix rows
              // correspond to shape functions which are in the lexicographic
              // order.
              ky_shape_grad_matrix_table_for_common_vertex(
                TableIndices<2>(k, q)) =
                BEMTools::shape_grad_matrix_in_lexicographic_order<
                  dim,
                  spacedim,
                  RangeNumberType>(ky_fe, ky_unit_quad_points[q]);
            }
        }

      /**
       * Compute mapping shape function values and their derivatives in batch.
       *
       * \alert{Even though the internally generated polynomials in the
       * mapping object are in the tensor product order, the shape function
       * values and derivatives are still in the hierarchic order. This can be
       * verified by checking the source code of
       * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
       * (see
       * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
       *
       * However, this behavior is different from the documentation for the
       * function @p MappingQ< dim, spacedim >::InternalData::shape().}
       */
      for (unsigned int m = 0; m < mappings.size(); m++)
        {
          auto              &mapping_data = mappings[m]->get_data();
          const unsigned int mapping_n_shape_functions =
            mapping_data->n_shape_functions;
          // The numbering used for accessing shape function values and their
          // derivatives in the lexicographic order.
          const std::vector<unsigned int> &lexicographic_numbering =
            mappings[m]->get_lexicographic_numberings_for_shape_functions()[0];

          // Compute shape function values and their derivatives for kx.
          mapping_data->compute_shape_function_values(kx_unit_quad_points);
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
                {
                  kx_mapping_shape_value_table_for_common_vertex(
                    TableIndices<4>(k, q, m, s)) =
                    static_cast<RangeNumberType>(
                      mapping_data->shape(q, lexicographic_numbering[s]));

                  for (unsigned int d = 0; d < dim; d++)
                    {
                      kx_mapping_shape_grad_matrix_table_for_common_vertex(
                        TableIndices<3>(m, k, q))(s, d) =
                        static_cast<RangeNumberType>(mapping_data->derivative(
                          q, lexicographic_numbering[s])[d]);
                    }
                }
            }

          // Compute shape function values and their derivatives for ky.
          mapping_data->compute_shape_function_values(ky_unit_quad_points);
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
                {
                  ky_mapping_shape_value_table_for_common_vertex(
                    TableIndices<4>(k, q, m, s)) =
                    static_cast<RangeNumberType>(
                      mapping_data->shape(q, lexicographic_numbering[s]));

                  for (unsigned int d = 0; d < dim; d++)
                    {
                      ky_mapping_shape_grad_matrix_table_for_common_vertex(
                        TableIndices<3>(m, k, q))(s, d) =
                        static_cast<RangeNumberType>(mapping_data->derivative(
                          q, lexicographic_numbering[s])[d]);
                    }
                }
            }
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::shape_function_values_regular()
{
  const unsigned int                  kx_dofs_per_cell = kx_fe.dofs_per_cell;
  const unsigned int                  ky_dofs_per_cell = ky_fe.dofs_per_cell;
  [[maybe_unused]] const unsigned int max_mapping_n_shape_functions =
    mappings.back()->get_data()->n_shape_functions;
  const unsigned int n_q_points = quad_rule_for_regular.size();

  // Make assertion about the length for each dimension of the data table.
  AssertDimension(kx_shape_value_table_for_regular.size(0), kx_dofs_per_cell);
  AssertDimension(kx_shape_value_table_for_regular.size(1), n_q_points);

  AssertDimension(ky_shape_value_table_for_regular.size(0), ky_dofs_per_cell);
  AssertDimension(ky_shape_value_table_for_regular.size(1), n_q_points);

  AssertDimension(mapping_shape_value_table_for_regular.size(0), n_q_points);
  AssertDimension(mapping_shape_value_table_for_regular.size(1),
                  mappings.size());
  AssertDimension(mapping_shape_value_table_for_regular.size(2),
                  max_mapping_n_shape_functions);

  if (is_surface_curl_needed)
    {
      AssertDimension(kx_shape_grad_matrix_table_for_regular.size(0),
                      n_q_points);
      AssertDimension(ky_shape_grad_matrix_table_for_regular.size(0),
                      n_q_points);
    }

  AssertDimension(mapping_shape_grad_matrix_table_for_regular.size(0),
                  mappings.size());
  AssertDimension(mapping_shape_grad_matrix_table_for_regular.size(1),
                  n_q_points);

  /**
   * Quadrature points in the Sauter's parametric space.
   */
  std::vector<Point<dim>> quad_points = quad_rule_for_regular.get_points();

  /**
   * Get the polynomial space inverse numbering for accessing the shape
   * functions in the lexicographic order.
   *
   * \alert{Here I have adopted an assumption that the finite elements are
   * based on tensor product polynomials.}
   */
  const FE_Poly_short &kx_fe_poly = dynamic_cast<const FE_Poly_short &>(kx_fe);
  const FE_Poly_short &ky_fe_poly = dynamic_cast<const FE_Poly_short &>(ky_fe);

  std::vector<unsigned int> kx_poly_space_inverse_numbering(
    kx_fe_poly.get_poly_space_numbering_inverse());
  std::vector<unsigned int> ky_poly_space_inverse_numbering(
    ky_fe_poly.get_poly_space_numbering_inverse());

  // Iterate over each quadrature point.
  for (unsigned int q = 0; q < n_q_points; q++)
    {
      /**
       * Iterate over each finite element shape function in the lexicographic
       * order in the unit cell and evaluate it at the current quadrature point.
       * N.B. For the regular case, there is no transformation from the Sauter
       * parametric space to the product unit cell space \f$\hat{K}_x \times
       * \hat{K}_y\f$.
       */
      for (unsigned int s = 0; s < kx_dofs_per_cell; s++)
        {
          kx_shape_value_table_for_regular(s, q) = static_cast<RangeNumberType>(
            kx_fe.shape_value(kx_poly_space_inverse_numbering[s],
                              quad_points[q]));
        }

      /**
       * Iterate over each finite element shape function in the lexicographic
       * order on the unit cell of \f$K_y\f$ and evaluate it at the current
       * quadrature point.
       */
      for (unsigned int s = 0; s < ky_dofs_per_cell; s++)
        {
          ky_shape_value_table_for_regular(s, q) = static_cast<RangeNumberType>(
            ky_fe.shape_value(ky_poly_space_inverse_numbering[s],
                              quad_points[q]));
        }

      if (is_surface_curl_needed)
        {
          // Calculate the Jacobian matrix evaluated at the quadrature point in
          // \f$K_x\f$. Matrix rows correspond to shape functions which are in
          // the lexicographic order.
          kx_shape_grad_matrix_table_for_regular(q) =
            BEMTools::shape_grad_matrix_in_lexicographic_order<dim,
                                                               spacedim,
                                                               RangeNumberType>(
              kx_fe, quad_points[q]);
          // Calculate the Jacobian matrix evaluated at the quadrature point in
          // \f$K_y\f$. Matrix rows correspond to shape functions which are in
          // the lexicographic order.
          ky_shape_grad_matrix_table_for_regular(q) =
            BEMTools::shape_grad_matrix_in_lexicographic_order<dim,
                                                               spacedim,
                                                               RangeNumberType>(
              ky_fe, quad_points[q]);
        }
    }

  /**
   * Compute mapping shape function values and their derivatives in batch.
   *
   * \alert{Even though the internally generated polynomials in the
   * mapping object are in the tensor product order, the shape function
   * values and derivatives are still in the hierarchic order. This can be
   * verified by checking the source code of
   * @p MappingQ<dim, spacedim>::InternalData::compute_shape_function_values.
   * (see
   * http://localhost/dealii-9.4.1-doc/mapping__q_8cc_source.html#l00271)
   *
   * However, this behavior is different from the documentation for the
   * function @p MappingQ< dim, spacedim >::InternalData::shape().}
   */
  // Initialize the internal data in the mapping objects.
  resize_internal_data_in_mappings(n_q_points);
  for (unsigned int m = 0; m < mappings.size(); m++)
    {
      auto              &mapping_data = mappings[m]->get_data();
      const unsigned int mapping_n_shape_functions =
        mapping_data->n_shape_functions;
      // The numbering used for accessing shape function values and their
      // derivatives in the lexicographic order.
      const std::vector<unsigned int> &lexicographic_numbering =
        mappings[m]->get_lexicographic_numberings_for_shape_functions()[0];

      // Compute mapping shape function values and their derivatives.
      mapping_data->compute_shape_function_values(quad_points);
      for (unsigned int q = 0; q < n_q_points; q++)
        {
          for (unsigned int s = 0; s < mapping_n_shape_functions; s++)
            {
              mapping_shape_value_table_for_regular(q, m, s) =
                static_cast<RangeNumberType>(
                  mapping_data->shape(q, lexicographic_numbering[s]));

              for (unsigned int d = 0; d < dim; d++)
                {
                  mapping_shape_grad_matrix_table_for_regular(m, q)(s, d) =
                    static_cast<RangeNumberType>(
                      mapping_data->derivative(q,
                                               lexicographic_numbering[s])[d]);
                }
            }
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
template <typename SurfaceNormalDetector>
void
BEMValues<dim, spacedim, RangeNumberType>::
  compute_bilinear_form_cell_values_for_regular(
    const Triangulation<dim, spacedim> &tria,
    const std::vector<unsigned int>    &mapping_indices,
    const SurfaceNormalDetector        &normal_detector)
{
  const types::global_cell_index n_cells       = tria.n_active_cells();
  const unsigned int             n_quad_points = quad_rule_for_regular.size();

  JxW_at_quad_points_for_regular.reinit(n_cells, n_quad_points);
  normals_at_quad_points_for_regular.reinit(n_cells, n_quad_points);
  quad_points_for_regular.reinit(n_cells, n_quad_points);

  if (is_surface_curl_needed)
    {
      kx_shape_curls_at_quad_points_for_regular.reinit(n_cells,
                                                       kx_fe.dofs_per_cell,
                                                       n_quad_points);
      ky_shape_curls_at_quad_points_for_regular.reinit(n_cells,
                                                       ky_fe.dofs_per_cell,
                                                       n_quad_points);
    }

  for (const auto &cell : tria.active_cell_iterators())
    {
      types::global_cell_index cell_index_global = cell->active_cell_index();
      const unsigned int mapping_index = mapping_indices[cell_index_global];
      const unsigned int mapping_n_shape_functions =
        HierBEM::PlatformShared::Utilities::fixed_power<2>(mapping_index + 2);

      for (unsigned int quad_no = 0; quad_no < n_quad_points; quad_no++)
        {
          compute_cell_values_at_a_quad_point_for_regular(
            quad_no,
            cell_index_global,
            cell_index_global,
            normal_detector.is_normal_vector_inward(cell->material_id()),
            mapping_index,
            mapping_n_shape_functions);
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
template <typename SurfaceNormalDetector>
void
BEMValues<dim, spacedim, RangeNumberType>::
  compute_bilinear_form_cell_values_for_regular(
    const std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>
                                                &cell_iterator_ptrs,
    const std::vector<types::global_cell_index> &local_to_global_cell_index_map,
    const std::vector<unsigned int>             &mapping_indices,
    const SurfaceNormalDetector                 &normal_detector)
{
  const types::global_cell_index n_cells =
    local_to_global_cell_index_map.size();
  AssertDimension(n_cells, cell_iterator_ptrs.size());
  const unsigned int n_quad_points = quad_rule_for_regular.size();

  JxW_at_quad_points_for_regular.reinit(n_cells, n_quad_points);
  normals_at_quad_points_for_regular.reinit(n_cells, n_quad_points);
  quad_points_for_regular.reinit(n_cells, n_quad_points);

  if (is_surface_curl_needed)
    {
      kx_shape_curls_at_quad_points_for_regular.reinit(n_cells,
                                                       kx_fe.dofs_per_cell,
                                                       n_quad_points);
      ky_shape_curls_at_quad_points_for_regular.reinit(n_cells,
                                                       ky_fe.dofs_per_cell,
                                                       n_quad_points);
    }

  for (types::global_cell_index cell_index_local = 0;
       cell_index_local < n_cells;
       cell_index_local++)
    {
      const typename DoFHandler<dim, spacedim>::cell_iterator cell =
        *cell_iterator_ptrs[cell_index_local];
      types::global_cell_index cell_index_global =
        local_to_global_cell_index_map[cell_index_local];
      const unsigned int mapping_index = mapping_indices[cell_index_global];
      const unsigned int mapping_n_shape_functions =
        HierBEM::PlatformShared::Utilities::fixed_power<2>(mapping_index + 2);

      for (unsigned int quad_no = 0; quad_no < n_quad_points; quad_no++)
        {
          compute_cell_values_at_a_quad_point_for_regular(
            quad_no,
            cell_index_local,
            cell_index_global,
            normal_detector.is_normal_vector_inward(cell->material_id()),
            mapping_index,
            mapping_n_shape_functions);
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
void
BEMValues<dim, spacedim, RangeNumberType>::
  compute_cell_values_at_a_quad_point_for_regular(
    const unsigned int             quad_no,
    const types::global_cell_index cell_index_local,
    const types::global_cell_index cell_index_global,
    const bool                     cell_normals_inward_flag,
    const unsigned int             mapping_index,
    const unsigned int             mapping_n_shape_functions)
{
  const RangeNumberType quad_weight =
    static_cast<RangeNumberType>(quad_rule_for_regular.get_weights()[quad_no]);

  BEMTools::PlatformShared::
    transform_quad_point_from_unit_to_permuted_real_cell(
      &mapping_shape_value_table_for_regular(quad_no, mapping_index, 0),
      &mapping_support_point_table(cell_index_global, 0),
      mapping_n_shape_functions,
      quad_points_for_regular(cell_index_local, quad_no));

  if (is_surface_curl_needed)
    {
      // Covariant matrix is computed on the stack without storing into
      // data tables.
      RangeNumberType covariant_matrix_data[spacedim * dim];
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        covariant_matrix(covariant_matrix_data, spacedim, dim);

      JxW_at_quad_points_for_regular(cell_index_local, quad_no) =
        BEMTools::PlatformShared::
          surface_jacobian_det_normal_vector_and_covariant(
            &mapping_shape_grad_matrix_table_for_regular(mapping_index, quad_no)
               .begin()
               ->value(),
            &mapping_support_point_table(cell_index_global, 0),
            mapping_n_shape_functions,
            normals_at_quad_points_for_regular(cell_index_local, quad_no),
            covariant_matrix,
            cell_normals_inward_flag) *
        quad_weight;

      // Iterate over each shape function in the finite element for
      // Kx.
      for (unsigned int fe_shape_index = 0;
           fe_shape_index < kx_fe.dofs_per_cell;
           fe_shape_index++)
        {
          BEMTools::PlatformShared::
            surface_curl<dim, spacedim, RangeNumberType>(
              fe_shape_index,
              &kx_shape_grad_matrix_table_for_regular(quad_no).begin()->value(),
              kx_fe.dofs_per_cell,
              covariant_matrix,
              normals_at_quad_points_for_regular(cell_index_local, quad_no),
              kx_shape_curls_at_quad_points_for_regular(cell_index_local,
                                                        fe_shape_index,
                                                        quad_no));
        }

      // Iterate over each shape function in the finite element for
      // Ky.
      for (unsigned int fe_shape_index = 0;
           fe_shape_index < ky_fe.dofs_per_cell;
           fe_shape_index++)
        {
          BEMTools::PlatformShared::
            surface_curl<dim, spacedim, RangeNumberType>(
              fe_shape_index,
              &ky_shape_grad_matrix_table_for_regular(quad_no).begin()->value(),
              ky_fe.dofs_per_cell,
              covariant_matrix,
              normals_at_quad_points_for_regular(cell_index_local, quad_no),
              ky_shape_curls_at_quad_points_for_regular(cell_index_local,
                                                        fe_shape_index,
                                                        quad_no));
        }
    }
  else
    {
      JxW_at_quad_points_for_regular(cell_index_local, quad_no) =
        BEMTools::PlatformShared::surface_jacobian_det_and_normal_vector(
          &mapping_shape_grad_matrix_table_for_regular(mapping_index, quad_no)
             .begin()
             ->value(),
          &mapping_support_point_table(cell_index_global, 0),
          mapping_n_shape_functions,
          normals_at_quad_points_for_regular(cell_index_local, quad_no),
          cell_normals_inward_flag) *
        quad_weight;
    }
}


/**
 * Structure holding cell-wise local matrix data and DoF indices, which is
 * used for SMP parallel computation of the scaled FEM mass matrix.
 */
template <int dim, int spacedim, typename RangeNumberType = double>
struct CellWiseCopyDataForMassMatrix
{
  LAPACKFullMatrixExt<RangeNumberType> local_matrix;
  // N.B. Memory should be preallocated for this vector before calling
  // <code>get_dof_indices</code>.
  std::vector<types::global_dof_index> local_dof_indices_for_test_space;
  std::vector<types::global_dof_index> local_dof_indices_for_trial_space;

  /**
   * Constructor. Allocate memory for internal members.
   *
   * @param fe_for_test_space
   * @param fe_for_trial_space
   */
  CellWiseCopyDataForMassMatrix(
    const FiniteElement<dim, spacedim> &fe_for_test_space,
    const FiniteElement<dim, spacedim> &fe_for_trial_space)
    : local_matrix(fe_for_test_space.dofs_per_cell,
                   fe_for_trial_space.dofs_per_cell)
    , local_dof_indices_for_test_space(fe_for_test_space.dofs_per_cell)
    , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
  {}

  /**
   * Copy constructor
   *
   * @param copy_data
   */
  CellWiseCopyDataForMassMatrix(
    const CellWiseCopyDataForMassMatrix<dim, spacedim, RangeNumberType>
      &copy_data)
    : local_matrix(copy_data.local_matrix)
    , local_dof_indices_for_test_space(
        copy_data.local_dof_indices_for_test_space)
    , local_dof_indices_for_trial_space(
        copy_data.local_dof_indices_for_trial_space)
  {}
};


template <int dim, int spacedim, typename RangeNumberType = double>
struct CellWiseCopyDataForMassMatrixVmult
{
  LAPACKFullMatrixExt<RangeNumberType> local_matrix;
  Vector<RangeNumberType>              local_u, local_v;

  // N.B. Memory should be preallocated for this vector before calling
  // <code>get_dof_indices</code>.
  std::vector<types::global_dof_index> local_dof_indices_for_test_space;
  std::vector<types::global_dof_index> local_dof_indices_for_trial_space;

  /**
   * Constructor. Allocate memory for internal members.
   *
   * @param fe_for_test_space
   * @param fe_for_trial_space
   */
  CellWiseCopyDataForMassMatrixVmult(
    const FiniteElement<dim, spacedim> &fe_for_test_space,
    const FiniteElement<dim, spacedim> &fe_for_trial_space)
    : local_matrix(fe_for_test_space.dofs_per_cell,
                   fe_for_trial_space.dofs_per_cell)
    , local_u(fe_for_test_space.dofs_per_cell)
    , local_v(fe_for_trial_space.dofs_per_cell)
    , local_dof_indices_for_test_space(fe_for_test_space.dofs_per_cell)
    , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
  {}

  /**
   * Copy constructor
   *
   * @param copy_data
   */
  CellWiseCopyDataForMassMatrixVmult(
    const CellWiseCopyDataForMassMatrix<dim, spacedim, RangeNumberType>
      &copy_data)
    : local_matrix(copy_data.local_matrix)
    , local_u(copy_data.local_u)
    , local_v(copy_data.local_v)
    , local_dof_indices_for_test_space(
        copy_data.local_dof_indices_for_test_space)
    , local_dof_indices_for_trial_space(
        copy_data.local_dof_indices_for_trial_space)
  {}
};


/**
 * Structure holding temporary data which are needed for cell-wise
 * integration, such as for the scaled mass matrix term \f$(v, \alpha \cdot
 * u)\f$.
 */
template <int dim, int spacedim>
struct CellWiseScratchDataForMassMatrix
{
  FEValues<dim, spacedim> fe_values_for_test_space;
  FEValues<dim, spacedim> fe_values_for_trial_space;

  /**
   * Constructor
   *
   * @param fe_for_test_space
   * @param fe_for_trial_space
   * @param quadrature
   * @param update_flags
   */
  CellWiseScratchDataForMassMatrix(
    const FiniteElement<dim, spacedim> &fe_for_test_space,
    const FiniteElement<dim, spacedim> &fe_for_trial_space,
    const Quadrature<dim>              &quadrature,
    const UpdateFlags                   update_flags)
    : fe_values_for_test_space(fe_for_test_space, quadrature, update_flags)
    , fe_values_for_trial_space(fe_for_trial_space, quadrature, update_flags)
  {}


  /**
   * Copy constructor. Because <code>FEValues</code> is neither copyable nor
   * has it copy constructor, this copy constructor is mandatory for
   * replication into each task.
   *
   * @param scratch_data
   */
  CellWiseScratchDataForMassMatrix(
    const CellWiseScratchDataForMassMatrix<dim, spacedim> &scratch_data)
    : fe_values_for_test_space(
        scratch_data.fe_values_for_test_space.get_fe(),
        scratch_data.fe_values_for_test_space.get_quadrature(),
        scratch_data.fe_values_for_test_space.get_update_flags())
    , fe_values_for_trial_space(
        scratch_data.fe_values_for_trial_space.get_fe(),
        scratch_data.fe_values_for_trial_space.get_quadrature(),
        scratch_data.fe_values_for_trial_space.get_update_flags())
  {}
};


/**
 * @brief Scratch data on each cell which is used for evaluation of the
 potential generated by layer charges.
 * @tparam dim
 * @tparam spacedim

 */
template <int dim, int spacedim>
struct CellWiseScratchDataForPotentialEval
{
  FEValues<dim, spacedim> fe_values_for_trial_space;

  CellWiseScratchDataForPotentialEval(
    const FiniteElement<dim, spacedim> &fe_for_trial_space,
    const Quadrature<dim>              &quadrature,
    const UpdateFlags                   update_flags)
    : fe_values_for_trial_space(fe_for_trial_space, quadrature, update_flags)
  {}

  /**
   * Copy constructor
   */
  CellWiseScratchDataForPotentialEval(
    const CellWiseScratchDataForPotentialEval<dim, spacedim> &scratch_data)
    : fe_values_for_trial_space(
        scratch_data.fe_values_for_trial_space.get_fe(),
        scratch_data.fe_values_for_trial_space.get_quadrature(),
        scratch_data.fe_values_for_trial_space.get_update_flags())
  {}
};


/**
 * @brief Cell local assembled vector data which are used for evaluation of the
 * potential generated by layer charges.
 * @tparam RangeNumberType
 * @tparam dim
 * @tparam spacedim
 */
template <int dim, int spacedim, typename RangeNumberType = double>
struct CellWisePerTaskDataForPotentialEval
{
  Vector<RangeNumberType>              local_vector;
  std::vector<types::global_dof_index> local_dof_indices_for_trial_space;

  CellWisePerTaskDataForPotentialEval(
    const FiniteElement<dim, spacedim> &fe_for_trial_space)
    : local_vector(fe_for_trial_space.dofs_per_cell)
    , local_dof_indices_for_trial_space(fe_for_trial_space.dofs_per_cell)
  {}

  CellWisePerTaskDataForPotentialEval(
    const CellWisePerTaskDataForPotentialEval<dim, spacedim, RangeNumberType>
      &task_data)
    : local_vector(task_data.local_vector)
    , local_dof_indices_for_trial_space(
        task_data.local_dof_indices_for_trial_space)
  {}
};


/**
 * Base class for holding pair-cell-wise local matrix data and DoF indices. It
 * is also directly used for assembling BEM H-matrix near field blocks via
 * task buffer.
 */
template <int dim, int spacedim, typename KernelNumberType = double>
class PairCellWiseScratchDataBase
{
public:
  using FE_Poly_short = FE_Poly<dim, spacedim>;
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  /**
   * The intersection set of the vertex local indices for the two cells
   * \f$K_x\f$ and \f$K_y\f$.
   */
  std::vector<std::pair<unsigned int, unsigned int>>
    common_vertex_pair_local_indices;

  /**
   * Permuted list of mapping support points in the real cell \f$K_x\f$.
   *
   * This vector will be filled with values by permuting the support points in
   * default order according to the @p poly_space_numbering_inverse. It will
   * further be copied to the ring buffer on the device. Its size is
   * preallocated to the number of support points in the highest order
   * mapping. The number of its effective values is determined from the size
   * of @p poly_space_numbering_inverse.
   */
  std::vector<Point<spacedim, real_type>> kx_mapping_support_points_permuted;
  /**
   * Permuted list of mapping support points in the real cell \f$K_y\f$.
   *
   * This vector will be filled with values by permuting the support points in
   * default order according to the @p poly_space_numbering_inverse. It will
   * further be copied to the ring buffer on the device. Its size is
   * preallocated to the number of support points in the highest order
   * mapping. The number of its effective values is determined from the size
   * of @p poly_space_numbering_inverse.
   */
  std::vector<Point<spacedim, real_type>> ky_mapping_support_points_permuted;

  /**
   * The list of DoF indices in \f$K_x\f$ which are ordered in the
   * default DoF order. This is directly retrieved from the function
   * @p DoFHandler::cell_iterator::get_dof_indices.
   */
  std::vector<types::global_dof_index>
    kx_local_dof_indices_in_default_dof_order;
  /**
   * The list of DoF indices in \f$K_y\f$ which are ordered in the
   * default DoF order. This is directly retrieved from the function
   * @p DoFHandler::cell_iterator::get_dof_indices.
   */
  std::vector<types::global_dof_index>
    ky_local_dof_indices_in_default_dof_order;

  /**
   * The numbering used for accessing the list of DoFs in \f$K_x\f$ in the
   * lexicographic order, where the list of DoFs are stored in the default DoF
   * order.
   */
  std::vector<unsigned int> kx_fe_poly_space_numbering_inverse;
  /**
   * The numbering used for accessing the list of DoFs in \f$K_y\f$ in the
   * lexicographic order, where the list of DoFs are stored in the default DoF
   * order.
   */
  std::vector<unsigned int> ky_fe_poly_space_numbering_inverse;

  /**
   * The numbering used for accessing the list of support points and
   * associated DoF indices in \f$K_x\f$ in the lexicographic order by
   * starting from a specific vertex, where the list of support points and
   * associated DoF indices are stored in the default DoF order.
   *
   * \mynote{"By starting from a specific vertex" means:
   * 1. In the same panel case, this numbering is not used because the first
   * vertex is the starting point by default.
   * 2. In the common edge case, start from the vertex which is the starting
   * point of the common edge.
   * 3. In the common vertex case, start from the common vertex.
   * 4. In the regular panel case, same as the same panel case.}
   */
  std::vector<unsigned int> kx_local_dof_permutation;
  /**
   * The numbering used for accessing the list of support points and
   * associated DoF indices in \f$K_y\f$ in the lexicographic order or the
   * reversed lexicographic order by starting from a specific vertex, where
   * the list of support points and associated DoF indices are stored in the
   * default DoF order.
   *
   * \mynote{"By starting from a specific vertex" means:
   * 1. In the same panel case, this numbering is not used because the first
   * vertex is the starting point by default.
   * 2. In the common edge case, start from the vertex which is the starting
   * point of the common edge. And the list of support points and associated
   * DoF indices are accessed in the reversed lexicographic order. Then the
   * cell orientation is reversed and the calculated normal vector should be
   * negated.
   * 3. In the common vertex case, start from the common vertex. And the list
   * of support points and associated DoF indices are accessed in the
   * lexicographic order.
   * 4. In the regular panel case, same as the same panel case.}
   */
  std::vector<unsigned int> ky_local_dof_permutation;

  /**
   * Constructor
   *
   * @param kx_fe
   * @param ky_fe
   * @param mappings A list of pointers to @p MappingInfo objects from the 1st
   * order to the highest order mapping.
   */
  PairCellWiseScratchDataBase(
    const FiniteElement<dim, spacedim>              &kx_fe,
    const FiniteElement<dim, spacedim>              &ky_fe,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings)
    : common_vertex_pair_local_indices(0)
    , kx_mapping_support_points_permuted(
        mappings.back()->get_data()->n_shape_functions)
    , ky_mapping_support_points_permuted(
        mappings.back()->get_data()->n_shape_functions)
    , kx_local_dof_indices_in_default_dof_order(kx_fe.dofs_per_cell)
    , ky_local_dof_indices_in_default_dof_order(ky_fe.dofs_per_cell)
    , kx_fe_poly_space_numbering_inverse(kx_fe.dofs_per_cell)
    , ky_fe_poly_space_numbering_inverse(ky_fe.dofs_per_cell)
    , kx_local_dof_permutation(kx_fe.dofs_per_cell)
    , ky_local_dof_permutation(ky_fe.dofs_per_cell)
  {
    common_vertex_pair_local_indices.reserve(
      GeometryInfo<dim>::vertices_per_cell);

    // Polynomial space inverse numbering for recovering the lexicographic
    // order.
    const FE_Poly_short &kx_fe_poly =
      dynamic_cast<const FE_Poly_short &>(kx_fe);
    const FE_Poly_short &ky_fe_poly =
      dynamic_cast<const FE_Poly_short &>(ky_fe);

    kx_fe_poly_space_numbering_inverse =
      kx_fe_poly.get_poly_space_numbering_inverse();
    ky_fe_poly_space_numbering_inverse =
      ky_fe_poly.get_poly_space_numbering_inverse();
  }


  /**
   * Default copy constructor.
   */
  PairCellWiseScratchDataBase(
    const PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType> &) =
    default;
};


/**
 * Class holding pair-cell-wise local matrix data and DoF indices, which
 * is used for building BEM full matrices.
 */
template <int dim, int spacedim, typename KernelNumberType = double>
class PairCellWiseScratchDataForFullMatrix
  : public PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType>
{
public:
  using FE_Poly_short = FE_Poly<dim, spacedim>;
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  /**
   * Whether surface curls of finite element shape functions should be computed.
   */
  bool is_surface_curl_needed;

  /**
   * Jacobian (scaled by quadrature weight) from the unit cell to the real cell
   * \f$K_x\f$ for each \f$k_3\f$ term and at each quadrature point for the same
   * panel case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, real_type> kx_jacobians_same_panel;
  /**
   * Jacobian (scaled by quadrature weight) from the unit cell to the real cell
   * \f$K_x\f$ for each \f$k_3\f$ term and at each quadrature point for the
   * common edge case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, real_type> kx_jacobians_common_edge;
  /**
   * Jacobian (scaled by quadrature weight) from the unit cell to the real cell
   * \f$K_x\f$ for each \f$k_3\f$ term and at each quadrature point for the
   * common vertex case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, real_type> kx_jacobians_common_vertex;

  /**
   * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
   * the same panel case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, Tensor<1, spacedim, real_type>> kx_normals_same_panel;
  /**
   * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
   * the common edge case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, Tensor<1, spacedim, real_type>> kx_normals_common_edge;
  /**
   * Normal vector at each quadrature point in the real cell \f$K_x\f$ for
   * the common vertex case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, Tensor<1, spacedim, real_type>> kx_normals_common_vertex;

  /**
   * Surface curl vectors of finite element shape functions at each quadrature
   * point in the real cell \f$K_x\f$ for the same panel case.
   * Dim1: fe shape function index
   * Dim2: k3 index
   * Dim3: quad point index
   */
  Table<3, Tensor<1, spacedim, real_type>> kx_shape_curls_same_panel;

  /**
   * Surface curl vectors of finite element shape functions at each quadrature
   * point in the real cell \f$K_x\f$ for the common edge case.
   * Dim1: fe shape function index
   * Dim2: k3 index
   * Dim3: quad point index
   */
  Table<3, Tensor<1, spacedim, real_type>> kx_shape_curls_common_edge;

  /**
   * Surface curl vectors of finite element shape functions at each quadrature
   * point in the real cell \f$K_x\f$ for the common vertex case.
   * Dim1: fe shape function index
   * Dim2: k3 index
   * Dim3: quad point index
   */
  Table<3, Tensor<1, spacedim, real_type>> kx_shape_curls_common_vertex;

  /**
   * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
   * quadrature point for the same panel case. The first dimension is the
   * \f$k_3\f$ index.
   */
  Table<2, Point<spacedim, real_type>> kx_quad_points_same_panel;
  /**
   * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
   * quadrature point for the common edge case. The first dimension is the
   * \f$k_3\f$ index.
   */
  Table<2, Point<spacedim, real_type>> kx_quad_points_common_edge;
  /**
   * Coordinates in the real cell \f$K_x\f$ for each \f$k_3\f$ term and each
   * quadrature point for the common vertex case. The first dimension is the
   * \f$k_3\f$ index.
   */
  Table<2, Point<spacedim, real_type>> kx_quad_points_common_vertex;

  /**
   * Jacobian (scaled by quadrature weight) from the unit cell to the real cell
   * \f$K_y\f$ for each \f$k_3\f$ term and at each quadrature point for the same
   * panel case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, real_type> ky_jacobians_same_panel;
  /**
   * Jacobian (scaled by quadrature weight) from the unit cell to the real cell
   * \f$K_y\f$ for each \f$k_3\f$ term and at each quadrature point for the
   * common edge case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, real_type> ky_jacobians_common_edge;
  /**
   * Jacobian (scaled by quadrature weight) from the unit cell to the real cell
   * \f$K_y\f$ for each \f$k_3\f$ term and at each quadrature point for the
   * common vertex case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, real_type> ky_jacobians_common_vertex;

  /**
   * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
   * the same panel case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, Tensor<1, spacedim, real_type>> ky_normals_same_panel;
  /**
   * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
   * the common edge case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, Tensor<1, spacedim, real_type>> ky_normals_common_edge;
  /**
   * Normal vector at each quadrature point in the real cell \f$K_y\f$ for
   * the common vertex case. The first dimension is the \f$k_3\f$ index.
   */
  Table<2, Tensor<1, spacedim, real_type>> ky_normals_common_vertex;

  /**
   * Surface curl vectors of finite element shape functions at each quadrature
   * point in the real cell \f$K_y\f$ for the same panel case.
   * Dim1: fe shape function index
   * Dim2: k3 index
   * Dim3: quad point index
   */
  Table<3, Tensor<1, spacedim, real_type>> ky_shape_curls_same_panel;

  /**
   * Surface curl vectors of finite element shape functions at each quadrature
   * point in the real cell \f$K_y\f$ for the common edge case.
   * Dim1: fe shape function index
   * Dim2: k3 index
   * Dim3: quad point index
   */
  Table<3, Tensor<1, spacedim, real_type>> ky_shape_curls_common_edge;

  /**
   * Surface curl vectors of finite element shape functions at each quadrature
   * point in the real cell \f$K_y\f$ for the common vertex case.
   * Dim1: fe shape function index
   * Dim2: k3 index
   * Dim3: quad point index
   */
  Table<3, Tensor<1, spacedim, real_type>> ky_shape_curls_common_vertex;

  /**
   * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
   * quadrature point for the same panel case. The first dimension is the
   * \f$k_3\f$ index.
   */
  Table<2, Point<spacedim, real_type>> ky_quad_points_same_panel;
  /**
   * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
   * quadrature point for the common edge case. The first dimension is the
   * \f$k_3\f$ index.
   */
  Table<2, Point<spacedim, real_type>> ky_quad_points_common_edge;
  /**
   * Coordinates in the real cell \f$K_y\f$ for each \f$k_3\f$ term and each
   * quadrature point for the common vertex case. The first dimension is the
   * \f$k_3\f$ index.
   */
  Table<2, Point<spacedim, real_type>> ky_quad_points_common_vertex;

  /**
   * Constructor
   *
   * @param kx_fe
   * @param ky_fe
   * @param mappings A list of pointers to @p MappingInfo objects from the 1st
   * order to the highest order mapping.
   * @param bem_values
   */
  PairCellWiseScratchDataForFullMatrix(
    const FiniteElement<dim, spacedim>              &kx_fe,
    const FiniteElement<dim, spacedim>              &ky_fe,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const QGauss<dim * 2>                           &quad_rule_for_same_panel,
    const QGauss<dim * 2>                           &quad_rule_for_common_edge,
    const QGauss<dim * 2> &quad_rule_for_common_vertex,
    const bool             is_surface_curl_needed)
    : PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType>(kx_fe,
                                                                   ky_fe,
                                                                   mappings)
    , is_surface_curl_needed(is_surface_curl_needed)
    , kx_jacobians_same_panel(8, quad_rule_for_same_panel.size())
    , kx_jacobians_common_edge(6, quad_rule_for_common_edge.size())
    , kx_jacobians_common_vertex(4, quad_rule_for_common_vertex.size())
    , kx_normals_same_panel(8, quad_rule_for_same_panel.size())
    , kx_normals_common_edge(6, quad_rule_for_common_edge.size())
    , kx_normals_common_vertex(4, quad_rule_for_common_vertex.size())
    , kx_quad_points_same_panel(8, quad_rule_for_same_panel.size())
    , kx_quad_points_common_edge(6, quad_rule_for_common_edge.size())
    , kx_quad_points_common_vertex(4, quad_rule_for_common_vertex.size())
    , ky_jacobians_same_panel(8, quad_rule_for_same_panel.size())
    , ky_jacobians_common_edge(6, quad_rule_for_common_edge.size())
    , ky_jacobians_common_vertex(4, quad_rule_for_common_vertex.size())
    , ky_normals_same_panel(8, quad_rule_for_same_panel.size())
    , ky_normals_common_edge(6, quad_rule_for_common_edge.size())
    , ky_normals_common_vertex(4, quad_rule_for_common_vertex.size())
    , ky_quad_points_same_panel(8, quad_rule_for_same_panel.size())
    , ky_quad_points_common_edge(6, quad_rule_for_common_edge.size())
    , ky_quad_points_common_vertex(4, quad_rule_for_common_vertex.size())
  {
    if (is_surface_curl_needed)
      {
        kx_shape_curls_same_panel.reinit(kx_fe.dofs_per_cell,
                                         8,
                                         quad_rule_for_same_panel.size());
        kx_shape_curls_common_edge.reinit(kx_fe.dofs_per_cell,
                                          6,
                                          quad_rule_for_common_edge.size());
        kx_shape_curls_common_vertex.reinit(kx_fe.dofs_per_cell,
                                            4,
                                            quad_rule_for_common_vertex.size());
        ky_shape_curls_same_panel.reinit(ky_fe.dofs_per_cell,
                                         8,
                                         quad_rule_for_same_panel.size());
        ky_shape_curls_common_edge.reinit(ky_fe.dofs_per_cell,
                                          6,
                                          quad_rule_for_common_edge.size());
        ky_shape_curls_common_vertex.reinit(ky_fe.dofs_per_cell,
                                            4,
                                            quad_rule_for_common_vertex.size());
      }
  }


  /**
   * Default copy constructor.
   */
  PairCellWiseScratchDataForFullMatrix(
    const PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
      &) = default;
};


/**
 * Class holding pair-cell-wise local matrix data and DoF indices, which
 * is used for building BEM H-matrices without using the producer-consumer
 * model.
 */
template <int dim, int spacedim, typename KernelNumberType = double>
class PairCellWiseScratchDataForHMatrix
  : public PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType>
{
public:
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  cudaStream_t cuda_stream_handle;

  /**
   * Buffer holding quadrature results accumulated from all thread blocks used
   * by @p ApplyQuadratureUsingBEMValues{NeighboringType}.
   */
  KernelNumberType *quad_values_in_thread_blocks;

  /**
   * Size of the array @p quad_values_in_thread_blocks, which should be allocated
   * with a size larger than the maximum possible number of thread blocks used
   * by @p ApplyQuadratureUsingBEMValues{NeighboringType}.
   */
  unsigned int n_quad_values_in_thread_blocks;

  PairCellWiseScratchDataForHMatrix(
    const FiniteElement<dim, spacedim>              &kx_fe,
    const FiniteElement<dim, spacedim>              &ky_fe,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const unsigned int n_quad_values_in_thread_blocks = 100)
    : PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType>(kx_fe,
                                                                   ky_fe,
                                                                   mappings)
    , cuda_stream_handle(0)
    , quad_values_in_thread_blocks(nullptr)
    , n_quad_values_in_thread_blocks(n_quad_values_in_thread_blocks)
  {
    AssertCuda(cudaStreamCreate(&cuda_stream_handle));

    /**
     * @internal Register host memory for asynchronous transfer to the device.
     */
    AssertCuda(
      cudaHostRegister((void *)this->kx_mapping_support_points_permuted.data(),
                       this->kx_mapping_support_points_permuted.size() *
                         sizeof(Point<spacedim, real_type>),
                       cudaHostRegisterDefault));

    AssertCuda(
      cudaHostRegister((void *)this->ky_mapping_support_points_permuted.data(),
                       this->ky_mapping_support_points_permuted.size() *
                         sizeof(Point<spacedim, real_type>),
                       cudaHostRegisterDefault));

    AssertCuda(cudaMallocHost((void **)&quad_values_in_thread_blocks,
                              n_quad_values_in_thread_blocks *
                                sizeof(KernelNumberType)));
  }

  PairCellWiseScratchDataForHMatrix(
    const PairCellWiseScratchDataForHMatrix<dim, spacedim, KernelNumberType>
      &) = default;

  void
  release()
  {
    AssertCuda(cudaStreamDestroy(cuda_stream_handle));

    AssertCuda(cudaHostUnregister(
      (void *)this->kx_mapping_support_points_permuted.data()));

    AssertCuda(cudaHostUnregister(
      (void *)this->ky_mapping_support_points_permuted.data()));

    if (quad_values_in_thread_blocks != nullptr)
      AssertCuda(cudaFreeHost(quad_values_in_thread_blocks));
  }
};


/**
 * Class holding pair-cell-wise local matrix data and DoF indices, which
 * is used for building BEM H-matrix far field blocks via task buffer.
 */
template <int dim, int spacedim, typename KernelNumberType = double>
class PairCellWiseScratchDataForHMatrixFarField
  : public PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType>
{
public:
  cudaStream_t cuda_stream_handle;

  PairCellWiseScratchDataForHMatrixFarField(
    const FiniteElement<dim, spacedim>              &kx_fe,
    const FiniteElement<dim, spacedim>              &ky_fe,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings)
    : PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType>(kx_fe,
                                                                   ky_fe,
                                                                   mappings)
    , cuda_stream_handle(0)
  {
    AssertCuda(cudaStreamCreate(&cuda_stream_handle));
  }

  PairCellWiseScratchDataForHMatrixFarField(
    const PairCellWiseScratchDataForHMatrixFarField<dim,
                                                    spacedim,
                                                    KernelNumberType> &) =
    default;

  ~PairCellWiseScratchDataForHMatrixFarField()
  {
    AssertCuda(cudaStreamDestroy(cuda_stream_handle));
  }
};


template <int dim, int spacedim>
class PairCellWisePerTaskData
{
public:
  /**
   * Permuted list of DoF indices in the cell \f$K_x\f$, each element of
   * which is associated with the corresponding element in
   * @p PairCellWiseScratchData::kx_support_points_permuted.
   */
  std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
  /**
   * Permuted list of DoF indices in the cell \f$K_y\f$, each element of
   * which is associated with the corresponding element in
   * @p PairCellWiseScratchData::ky_support_points_permuted.
   */
  std::vector<types::global_dof_index> ky_local_dof_indices_permuted;

  /**
   * Constructor
   *
   * @param kx_fe
   * @param ky_fe
   */
  PairCellWisePerTaskData(const FiniteElement<dim, spacedim> &kx_fe,
                          const FiniteElement<dim, spacedim> &ky_fe)
    : kx_local_dof_indices_permuted(kx_fe.dofs_per_cell)
    , ky_local_dof_indices_permuted(ky_fe.dofs_per_cell)
  {}


  /**
   * Default copy constructor.
   */
  PairCellWisePerTaskData(const PairCellWisePerTaskData<dim, spacedim> &) =
    default;
};


template <int dim, int spacedim, typename KernelNumberType = double>
class PairCellWisePerTaskDataForFullMatrix
  : public PairCellWisePerTaskData<dim, spacedim>
{
public:
  /**
   * Local matrix for the pair of cells to be assembled into the global full
   * matrix representation of the boundary integral operator.
   *
   * \comment{Therefore, this data field is only defined for verification.}
   */
  LAPACKFullMatrixExt<KernelNumberType> local_pair_cell_matrix;

  /**
   * Constructor
   *
   * @param kx_fe
   * @param ky_fe
   */
  PairCellWisePerTaskDataForFullMatrix(
    const FiniteElement<dim, spacedim> &kx_fe,
    const FiniteElement<dim, spacedim> &ky_fe)
    : PairCellWisePerTaskData<dim, spacedim>(kx_fe, ky_fe)
    , local_pair_cell_matrix(kx_fe.dofs_per_cell, ky_fe.dofs_per_cell)
  {}


  /**
   * Default copy constructor.
   */
  PairCellWisePerTaskDataForFullMatrix(
    const PairCellWisePerTaskDataForFullMatrix<dim, spacedim, KernelNumberType>
      &) = default;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_BEM_VALUES_H_
