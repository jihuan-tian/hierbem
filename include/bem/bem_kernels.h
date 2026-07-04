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
 * @file bem_kernels.h
 * @brief Definition of BEM kernel function classes for pullbacks to the product
 * unit cell space and Sauter parametric space.
 *
 * @date 2022-03-04
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_BEM_BEM_KERNELS_H_
#define HIERBEM_INCLUDE_BEM_BEM_KERNELS_H_

#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <cassert>

#include "bem/bem_values.h"
#include "bem/cell_neighboring_type.h"
#include "bem/types.h"
#include "config.h"
#include "platform_shared/tensor.h"
#include "platform_shared/utilities.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Kernel function pulled back to the unit cell.
 *
 * \mynote{1. The unit cell has the manifold dimension @p dim.
 * 2. Here we use the "template template" technique to define the template
 * parameter @p KernelFunctionType. Its first template parameter is the
 * space dimension and the second parameter is the value type.}
 */
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType = double>
class KernelPulledbackToUnitCell
{
public:
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;
  const unsigned int n_components;

  /**
   * Constructor on the device.
   *
   * @param kernel_function Kernel function for a boundary integral operator.
   * @param cell_neighboring_type Cell neighboring type.
   * @param kx_dof_index Cell local DoF index for accessing the list of DoFs
   * in the lexicographic order in \f$K_x\f$.
   * @param ky_dof_index Cell local DoF index for accessing the list of DoFs
   * in the lexicographic order or reversed lexicographic order in \f$K_y\f$.
   */

  KernelPulledbackToUnitCell(
    const KernelFunctionType<spacedim, KernelNumberType> &kernel_function,
    const CellNeighboringType                            &cell_neighboring_type,
    const unsigned int                                    kx_dof_index = 0,
    const unsigned int                                    ky_dof_index = 0);

  /**
   * Destructor
   *
   * \mynote{Since this class has only references to other objects as its
   * members and their memory is not managed by this class, the destructor
   * provided by the compiler is adopted.}
   */
  ~KernelPulledbackToUnitCell();

  /**
   * Assignment operator
   */
  KernelPulledbackToUnitCell &
  operator=(const KernelPulledbackToUnitCell &f);

  /**
   * Set the cell local DoF index for \f$K_x\f$.
   *
   * @param kx_dof_index
   */
  void
  set_kx_dof_index(const unsigned int kx_dof_index);

  /**
   * Set the cell local DoF index for \f$K_y\f$.
   *
   * @param ky_dof_index
   */
  void
  set_ky_dof_index(const unsigned int ky_dof_index);

  /**
   * Evaluate the kernel function.
   *
   * This version is used for same panel, common edge and common vertex cell
   * neighboring types.
   *
   * @param k3_index
   * @param quad_no
   * @param bem_values
   * @param scratch_data
   * @param component
   * @return
   */
  KernelNumberType
  value(
    const unsigned int                         k3_index,
    const unsigned int                         quad_no,
    const BEMValues<dim, spacedim, real_type> &bem_values,
    const PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                      &scratch_data,
    const unsigned int component = 0) const;

  /**
   * Evaluate the kernel function.
   *
   * This version is only used for the regular cell neighboring type.
   *
   * @param kx_cell_index_local Cell index for \f$K_x\f$, which is used to
   * access
   * precomputed data tables in @p bem_values.
   * @param ky_cell_index_local Cell index for \f$K_y\f$, which is used to
   * access
   * precomputed data tables in @p bem_values.
   * @param kx_quad_no Quadrature point index of the 2D quadrature object for
   * \f$K_x\f$
   * @param ky_quad_no Quadrature point index of the 2D quadrature object for
   * \f$K_y\f$
   * @param bem_values @p BEMValues object containing precomputed data tables
   * @param component Component index
   * @return
   */
  KernelNumberType
  value(const types::global_cell_index             kx_cell_index_local,
        const types::global_cell_index             ky_cell_index_local,
        const unsigned int                         kx_quad_no,
        const unsigned int                         ky_quad_no,
        const BEMValues<dim, spacedim, real_type> &bem_values,
        const unsigned int                         component = 0) const;

private:
  const KernelFunctionType<spacedim, KernelNumberType> &kernel_function;

  CellNeighboringType cell_neighboring_type;

  /**
   * Cell local DoF index for accessing the list of DoFs in the lexicographic
   * order in \f$K_x\f$.
   */
  unsigned int kx_dof_index;
  /**
   * Cell local DoF index for accessing the list of DoFs in the lexicographic
   * order or reversed lexicographic order in \f$K_y\f$.
   */
  unsigned int ky_dof_index;
};


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>

KernelPulledbackToUnitCell<dim,
                           spacedim,
                           KernelFunctionType,
                           KernelNumberType>::
  KernelPulledbackToUnitCell(
    const KernelFunctionType<spacedim, KernelNumberType> &kernel_function,
    const CellNeighboringType                            &cell_neighboring_type,
    const unsigned int                                    kx_dof_index,
    const unsigned int                                    ky_dof_index)
  : n_components(kernel_function.n_components)
  , kernel_function(kernel_function)
  , cell_neighboring_type(cell_neighboring_type)
  , kx_dof_index(kx_dof_index)
  , ky_dof_index(ky_dof_index)
{}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelPulledbackToUnitCell<dim,
                           spacedim,
                           KernelFunctionType,
                           KernelNumberType>::~KernelPulledbackToUnitCell() =
  default;


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelPulledbackToUnitCell<dim, spacedim, KernelFunctionType, KernelNumberType>
                       &
  KernelPulledbackToUnitCell<
                         dim,
                         spacedim,
                         KernelFunctionType,
                         KernelNumberType>::operator=(const KernelPulledbackToUnitCell &f)
{
  assert(n_components == f.n_components);

  kernel_function       = f.kernel_function;
  cell_neighboring_type = f.cell_neighboring_type;
  kx_dof_index          = f.kx_dof_index;
  ky_dof_index          = f.ky_dof_index;

  return *this;
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
void
KernelPulledbackToUnitCell<
  dim,
  spacedim,
  KernelFunctionType,
  KernelNumberType>::set_kx_dof_index(const unsigned int kx_dof_index)
{
  this->kx_dof_index = kx_dof_index;
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
void
KernelPulledbackToUnitCell<
  dim,
  spacedim,
  KernelFunctionType,
  KernelNumberType>::set_ky_dof_index(const unsigned int ky_dof_index)
{
  this->ky_dof_index = ky_dof_index;
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelNumberType
KernelPulledbackToUnitCell<dim,
                           spacedim,
                           KernelFunctionType,
                           KernelNumberType>::
  value(
    const unsigned int                         k3_index,
    const unsigned int                         quad_no,
    const BEMValues<dim, spacedim, real_type> &bem_values,
    const PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                      &scratch_data,
    const unsigned int component) const
{
  real_type                      kx_shape_value, ky_shape_value;
  Point<spacedim, real_type>     x, y;
  real_type                      Jx, Jy;
  Tensor<1, spacedim, real_type> nx, ny;
  Tensor<1, spacedim, real_type> kx_shape_curl, ky_shape_curl;

  switch (cell_neighboring_type)
    {
        case CellNeighboringType::SamePanel: {
          kx_shape_value =
            bem_values.kx_shape_value_table_for_same_panel(kx_dof_index,
                                                           k3_index,
                                                           quad_no);
          ky_shape_value =
            bem_values.ky_shape_value_table_for_same_panel(ky_dof_index,
                                                           k3_index,
                                                           quad_no);

          if (kernel_function.kernel_type == KernelType::HyperSingularRegular)
            {
              assert(bem_values.is_surface_curl_needed);
              assert(scratch_data.is_surface_curl_needed);

              kx_shape_curl =
                scratch_data.kx_shape_curls_same_panel(kx_dof_index,
                                                       k3_index,
                                                       quad_no);
              ky_shape_curl =
                scratch_data.ky_shape_curls_same_panel(ky_dof_index,
                                                       k3_index,
                                                       quad_no);
            }

          x  = scratch_data.kx_quad_points_same_panel(k3_index, quad_no);
          y  = scratch_data.ky_quad_points_same_panel(k3_index, quad_no);
          Jx = scratch_data.kx_jacobians_same_panel(k3_index, quad_no);
          Jy = scratch_data.ky_jacobians_same_panel(k3_index, quad_no);
          nx = scratch_data.kx_normals_same_panel(k3_index, quad_no);
          ny = scratch_data.ky_normals_same_panel(k3_index, quad_no);

          break;
        }
        case CellNeighboringType::CommonEdge: {
          kx_shape_value =
            bem_values.kx_shape_value_table_for_common_edge(kx_dof_index,
                                                            k3_index,
                                                            quad_no);
          ky_shape_value =
            bem_values.ky_shape_value_table_for_common_edge(ky_dof_index,
                                                            k3_index,
                                                            quad_no);

          if (kernel_function.kernel_type == KernelType::HyperSingularRegular)
            {
              assert(bem_values.is_surface_curl_needed);
              assert(scratch_data.is_surface_curl_needed);

              kx_shape_curl =
                scratch_data.kx_shape_curls_common_edge(kx_dof_index,
                                                        k3_index,
                                                        quad_no);
              ky_shape_curl =
                scratch_data.ky_shape_curls_common_edge(ky_dof_index,
                                                        k3_index,
                                                        quad_no);
            }

          x  = scratch_data.kx_quad_points_common_edge(k3_index, quad_no);
          y  = scratch_data.ky_quad_points_common_edge(k3_index, quad_no);
          Jx = scratch_data.kx_jacobians_common_edge(k3_index, quad_no);
          Jy = scratch_data.ky_jacobians_common_edge(k3_index, quad_no);
          nx = scratch_data.kx_normals_common_edge(k3_index, quad_no);
          ny = scratch_data.ky_normals_common_edge(k3_index, quad_no);

          break;
        }
        case CellNeighboringType::CommonVertex: {
          kx_shape_value =
            bem_values.kx_shape_value_table_for_common_vertex(kx_dof_index,
                                                              k3_index,
                                                              quad_no);
          ky_shape_value =
            bem_values.ky_shape_value_table_for_common_vertex(ky_dof_index,
                                                              k3_index,
                                                              quad_no);

          if (kernel_function.kernel_type == KernelType::HyperSingularRegular)
            {
              assert(bem_values.is_surface_curl_needed);
              assert(scratch_data.is_surface_curl_needed);

              kx_shape_curl =
                scratch_data.kx_shape_curls_common_vertex(kx_dof_index,
                                                          k3_index,
                                                          quad_no);
              ky_shape_curl =
                scratch_data.ky_shape_curls_common_vertex(ky_dof_index,
                                                          k3_index,
                                                          quad_no);
            }

          x  = scratch_data.kx_quad_points_common_vertex(k3_index, quad_no);
          y  = scratch_data.ky_quad_points_common_vertex(k3_index, quad_no);
          Jx = scratch_data.kx_jacobians_common_vertex(k3_index, quad_no);
          Jy = scratch_data.ky_jacobians_common_vertex(k3_index, quad_no);
          nx = scratch_data.kx_normals_common_vertex(k3_index, quad_no);
          ny = scratch_data.ky_normals_common_vertex(k3_index, quad_no);

          break;
        }
        default: {
          assert(false);
          kx_shape_value = 0.;
          ky_shape_value = 0.;
          Jx             = 0.;
          Jy             = 0.;
        }
    }

  if (kernel_function.kernel_type == KernelType::HyperSingularRegular)
    {
      return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
             PlatformShared::scalar_product(kx_shape_curl, ky_shape_curl);
    }
  else
    {
      return kernel_function.value(x, y, nx, ny, component) * Jx * Jy *
             kx_shape_value * ky_shape_value;
    }
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelNumberType
KernelPulledbackToUnitCell<dim,
                           spacedim,
                           KernelFunctionType,
                           KernelNumberType>::
  value(const types::global_cell_index             kx_cell_index_local,
        const types::global_cell_index             ky_cell_index_local,
        const unsigned int                         kx_quad_no,
        const unsigned int                         ky_quad_no,
        const BEMValues<dim, spacedim, real_type> &bem_values,
        const unsigned int                         component) const
{
  const Point<spacedim, real_type> x =
    bem_values.quad_points_for_regular(kx_cell_index_local, kx_quad_no);
  const Point<spacedim, real_type> y =
    bem_values.quad_points_for_regular(ky_cell_index_local, ky_quad_no);
  const real_type JxW_x =
    bem_values.JxW_at_quad_points_for_regular(kx_cell_index_local, kx_quad_no);
  const real_type JxW_y =
    bem_values.JxW_at_quad_points_for_regular(ky_cell_index_local, ky_quad_no);
  const Tensor<1, spacedim, real_type> nx =
    bem_values.normals_at_quad_points_for_regular(kx_cell_index_local,
                                                  kx_quad_no);
  const Tensor<1, spacedim, real_type> ny =
    bem_values.normals_at_quad_points_for_regular(ky_cell_index_local,
                                                  ky_quad_no);

  if (kernel_function.kernel_type == KernelType::HyperSingularRegular)
    {
      assert(bem_values.is_surface_curl_needed);

      const Tensor<1, spacedim, real_type> kx_shape_curl =
        bem_values.kx_shape_curls_at_quad_points_for_regular(
          kx_cell_index_local, kx_dof_index, kx_quad_no);
      const Tensor<1, spacedim, real_type> ky_shape_curl =
        bem_values.ky_shape_curls_at_quad_points_for_regular(
          ky_cell_index_local, ky_dof_index, ky_quad_no);

      return kernel_function.value(x, y, nx, ny, component) * JxW_x * JxW_y *
             PlatformShared::scalar_product(kx_shape_curl, ky_shape_curl);
    }
  else
    {
      const real_type kx_shape_value =
        bem_values.kx_shape_value_table_for_regular(kx_dof_index, kx_quad_no);
      const real_type ky_shape_value =
        bem_values.ky_shape_value_table_for_regular(ky_dof_index, ky_quad_no);

      return kernel_function.value(x, y, nx, ny, component) * JxW_x * JxW_y *
             kx_shape_value * ky_shape_value;
    }
}


/**
 * Class for pullback the kernel function on the product of two unit cells
 * to Sauter's parametric space.
 */
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType = double>
class KernelPulledbackToSauterSpace
{
public:
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;
  const unsigned int n_components;

  /**
   * Constructor with @p BEMValues
   *
   * @param kernel
   * @param cell_neighboring_type
   * @param bem_values
   */

  KernelPulledbackToSauterSpace(
    const KernelPulledbackToUnitCell<dim,
                                     spacedim,
                                     KernelFunctionType,
                                     KernelNumberType> &kernel,
    const CellNeighboringType                           cell_neighboring_type);

  /**
   * Destructor
   *
   * \mynote{Since this class has only references to other objects as its
   * members and their memory is not managed by this class, the destructor
   * provided by the compiler is adopted.}
   */
  ~KernelPulledbackToSauterSpace();

  /**
   * Assignment operator
   */
  KernelPulledbackToSauterSpace &
  operator=(const KernelPulledbackToSauterSpace &f);

  /**
   * Evaluate the pullback of the kernel function on the host on Sauter's
   * parametric space at the given quadrature point under the given 4D
   * quadrature rule.
   *
   * This version is used for same panel, common edge and common vertex cell
   * neighboring types.
   *
   * @param quad_no
   * @param bem_values
   * @param scratch_data
   * @param component
   * @return
   */
  KernelNumberType
  value(
    const unsigned int                         quad_no,
    const BEMValues<dim, spacedim, real_type> &bem_values,
    const PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                      &scratch_data,
    const unsigned int component = 0) const;

  /**
   * Evaluate the pullback of the kernel function on the host on Sauter's
   * parametric space at the given quadrature point formed by the tensor
   * product of two 2D quadrature objects.
   *
   * This version is only used for the regular cell neighboring type.
   *
   * @param kx_cell_index_local
   * @param ky_cell_index_local
   * @param kx_quad_no
   * @param ky_quad_no
   * @param bem_values
   * @param component
   * @return
   */
  KernelNumberType
  value(const types::global_cell_index             kx_cell_index_local,
        const types::global_cell_index             ky_cell_index_local,
        const unsigned int                         kx_quad_no,
        const unsigned int                         ky_quad_no,
        const BEMValues<dim, spacedim, real_type> &bem_values,
        const unsigned int                         component = 0) const;

  CellNeighboringType
  get_cell_neighboring_type() const;

private:
  const KernelPulledbackToUnitCell<dim,
                                   spacedim,
                                   KernelFunctionType,
                                   KernelNumberType> &kernel_on_unit_cell;
  CellNeighboringType                                 cell_neighboring_type;
};


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>

KernelPulledbackToSauterSpace<dim,
                              spacedim,
                              KernelFunctionType,
                              KernelNumberType>::
  KernelPulledbackToSauterSpace(
    const KernelPulledbackToUnitCell<dim,
                                     spacedim,
                                     KernelFunctionType,
                                     KernelNumberType> &kernel,
    const CellNeighboringType                           cell_neighboring_type)
  : n_components(kernel.n_components)
  , kernel_on_unit_cell(kernel)
  , cell_neighboring_type(cell_neighboring_type)
{}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelPulledbackToSauterSpace<
  dim,
  spacedim,
  KernelFunctionType,
  KernelNumberType>::~KernelPulledbackToSauterSpace() = default;


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelPulledbackToSauterSpace<dim,
                              spacedim,
                              KernelFunctionType,
                              KernelNumberType>                      &
KernelPulledbackToSauterSpace<
                       dim,
                       spacedim,
                       KernelFunctionType,
                       KernelNumberType>::operator=(const KernelPulledbackToSauterSpace &f)
{
  assert(n_components == f.n_components);

  kernel_on_unit_cell   = f.kernel_on_unit_cell;
  cell_neighboring_type = f.cell_neighboring_type;

  return *this;
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelNumberType
KernelPulledbackToSauterSpace<dim,
                              spacedim,
                              KernelFunctionType,
                              KernelNumberType>::
  value(
    const unsigned int                         quad_no,
    const BEMValues<dim, spacedim, real_type> &bem_values,
    const PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                      &scratch_data,
    const unsigned int component) const
{
  KernelNumberType kernel_value = KernelNumberType();

  switch (cell_neighboring_type)
    {
        case CellNeighboringType::SamePanel: {
          // Current point in the Sauter's parametric space, at which the
          // pulled back kernel function is to be evaluated.
          const Point<dim * 2> &p =
            bem_values.quad_rule_for_same_panel.point(quad_no);
          real_type jacobian_det =
            static_cast<real_type>(p(0) * (1 - p(0)) * (1 - p(0) * p(1)));

          for (unsigned int k3_index = 0; k3_index < 8; k3_index++)
            {
              kernel_value += kernel_on_unit_cell.value(
                k3_index, quad_no, bem_values, scratch_data, component);
            }

          kernel_value *= jacobian_det;

          break;
        }
        case CellNeighboringType::CommonEdge: {
          // Current point in the Sauter's parametric space, at which the
          // pulled back kernel function is to be evaluated.
          const Point<dim * 2> &p =
            bem_values.quad_rule_for_common_edge.point(quad_no);
          real_type jacobian_det1 =
            static_cast<real_type>(p(0) * p(0) * (1 - p(0)));
          real_type jacobian_det2 =
            static_cast<real_type>(p(0) * p(0) * (1 - p(0) * p(1)));

          kernel_value =
            jacobian_det1 *
              (kernel_on_unit_cell.value(
                 0, quad_no, bem_values, scratch_data, component) +
               kernel_on_unit_cell.value(
                 1, quad_no, bem_values, scratch_data, component)) +
            jacobian_det2 *
              (kernel_on_unit_cell.value(
                 2, quad_no, bem_values, scratch_data, component) +
               kernel_on_unit_cell.value(
                 3, quad_no, bem_values, scratch_data, component) +
               kernel_on_unit_cell.value(
                 4, quad_no, bem_values, scratch_data, component) +
               kernel_on_unit_cell.value(
                 5, quad_no, bem_values, scratch_data, component));

          break;
        }
        case CellNeighboringType::CommonVertex: {
          // Current point in the Sauter's parametric space, at which the
          // pulled back kernel function is to be evaluated.
          const Point<dim * 2> &p =
            bem_values.quad_rule_for_common_vertex.point(quad_no);
          real_type jacobian_det = static_cast<real_type>(
            PlatformShared::Utilities::fixed_power<3>(p(0)));

          for (unsigned int k3_index = 0; k3_index < 4; k3_index++)
            {
              kernel_value += kernel_on_unit_cell.value(
                k3_index, quad_no, bem_values, scratch_data, component);
            }

          kernel_value *= jacobian_det;

          break;
        }
      default:
        Assert(false, ExcInternalError());
    }

  return kernel_value;
}



template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelNumberType
KernelPulledbackToSauterSpace<dim,
                              spacedim,
                              KernelFunctionType,
                              KernelNumberType>::
  value(const types::global_cell_index             kx_cell_index_local,
        const types::global_cell_index             ky_cell_index_local,
        const unsigned int                         kx_quad_no,
        const unsigned int                         ky_quad_no,
        const BEMValues<dim, spacedim, real_type> &bem_values,
        const unsigned int                         component) const
{
  assert(cell_neighboring_type == CellNeighboringType::Regular);

  // There is no k3 terms in a normal Gaussian quadrature and there is no
  // coordinate transformation from the Sauter parametric space to the product
  // space of two cells for the regular case, so we directly evaluate the
  // kernel function on the product space of two unit cells.
  return kernel_on_unit_cell.value(kx_cell_index_local,
                                   ky_cell_index_local,
                                   kx_quad_no,
                                   ky_quad_no,
                                   bem_values,
                                   component);
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
CellNeighboringType
KernelPulledbackToSauterSpace<dim,
                              spacedim,
                              KernelFunctionType,
                              KernelNumberType>::get_cell_neighboring_type()
  const
{
  return cell_neighboring_type;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_BEM_KERNELS_H_
