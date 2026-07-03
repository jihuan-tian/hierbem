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
 * @file bem_tools.h
 * @brief Low level functions used in BEM.
 *
 * @date 2026-05-12
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_PLATFORM_SHARED_BEM_TOOLS_H_
#define HIERBEM_INCLUDE_PLATFORM_SHARED_BEM_TOOLS_H_

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <cassert>
#include <cmath>

#include "config.h"
#include "full_matrix_wrapper.h"
#include "tensor.h"
#include "vector_wrapper.h"

HBEM_NS_OPEN

namespace BEMTools
{
  namespace PlatformShared
  {
    /**
     * Calculate surface Jacobian determinant and normal vector based on the
     * gradient matrix of mapping shape functions provided at a quadrature
     * point.
     *
     * \mynote{N.B. The reversed lexicographic order appears for \f$K_y\f$
     * when the cell neighboring type is common edge. Then the calculated
     * normal vector \f$n_y\f$ has the opposite direction of the real one,
     * which should be negated in the subsequent calculation.}
     *
     * @param mapping_shape_grad_matrix_data Pointer to the gradient matrix data
     * (in the unit cell) of mapping shape functions at a quadrature point.
     * @param mapping_support_points_ptr Pointer to the list of mapping support
     * points in the real cell in the lexicographic order.
     * @param mapping_n_shape_functions Number of mapping shape functions
     * @param normal_vector [out] Normal vector at the quadrature point to be
     * computed
     * @param is_normal_vector_negated Whether the direction of the computed
     * normal vector should be negated. By default, it is false.
     * @return Jacobian determinant for the map from the unit cell to the real
     * cell
     */
    template <int spacedim, typename RangeNumberType = double>
    HBEM_ATTR_HOST HBEM_ATTR_DEV RangeNumberType
    surface_jacobian_det_and_normal_vector(
      const RangeNumberType                  *mapping_shape_grad_matrix_data,
      const Point<spacedim, RangeNumberType> *mapping_support_points_ptr,
      const unsigned int                      mapping_n_shape_functions,
      Tensor<1, spacedim, RangeNumberType>   &normal_vector,
      const bool                              is_normal_vector_negated = false)
    {
      // Currently, only @p spacedim=3 is supported.
      assert(spacedim == 3);

      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        mapping_shape_grad_matrix_at_quad_point(
          const_cast<RangeNumberType *>(mapping_shape_grad_matrix_data),
          mapping_n_shape_functions,
          2);

      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        mapping_support_point_coordinate_matrix(
          const_cast<RangeNumberType *>(
            reinterpret_cast<const RangeNumberType *>(
              mapping_support_points_ptr)),
          spacedim,
          mapping_n_shape_functions);

      // Compute the Jacobian matrix of the cell mapping.
      RangeNumberType jacobian_matrix_data[spacedim * 2];
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        jacobian_matrix(jacobian_matrix_data, spacedim, 2);
      mapping_support_point_coordinate_matrix.mmult(
        jacobian_matrix, mapping_shape_grad_matrix_at_quad_point);

      // Compute the Gramian matrix, i.e. metric tensor.
      RangeNumberType                                             G_data[4];
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType> G(G_data,
                                                                    2,
                                                                    2);
      jacobian_matrix.Tmmult(G, jacobian_matrix);

#ifdef __CUDA_ARCH__
      RangeNumberType surface_jacobian_det = ::sqrt(G.determinant2x2());
#else
      RangeNumberType surface_jacobian_det = std::sqrt(G.determinant2x2());
#endif

      if (is_normal_vector_negated)
        {
          normal_vector[0] =
            (jacobian_matrix_data[2] * jacobian_matrix_data[4] -
             jacobian_matrix_data[1] * jacobian_matrix_data[5]) /
            surface_jacobian_det;
          normal_vector[1] =
            (jacobian_matrix_data[0] * jacobian_matrix_data[5] -
             jacobian_matrix_data[2] * jacobian_matrix_data[3]) /
            surface_jacobian_det;
          normal_vector[2] =
            (jacobian_matrix_data[1] * jacobian_matrix_data[3] -
             jacobian_matrix_data[0] * jacobian_matrix_data[4]) /
            surface_jacobian_det;
        }
      else
        {
          normal_vector[0] =
            (jacobian_matrix_data[1] * jacobian_matrix_data[5] -
             jacobian_matrix_data[2] * jacobian_matrix_data[4]) /
            surface_jacobian_det;
          normal_vector[1] =
            (jacobian_matrix_data[2] * jacobian_matrix_data[3] -
             jacobian_matrix_data[0] * jacobian_matrix_data[5]) /
            surface_jacobian_det;
          normal_vector[2] =
            (jacobian_matrix_data[0] * jacobian_matrix_data[4] -
             jacobian_matrix_data[1] * jacobian_matrix_data[3]) /
            surface_jacobian_det;
        }

      return surface_jacobian_det;
    }


    /**
     * Calculate surface Jacobian determinant, normal vector and covariant
     * matrix based on the gradient matrix of mapping shape functions provided
     * at a quadrature point.
     *
     * @param mapping_shape_grad_matrix_data Pointer to the gradient matrix data
     * (in the unit cell) of mapping shape functions at a quadrature point.
     * @param mapping_support_points_ptr Pointer to the list of mapping support
     * points in the real cell in the lexicographic order.
     * @param mapping_n_shape_functions Number of mapping shape functions
     * @param normal_vector [out] Normal vector at the quadrature point to be
     * computed
     * @param covariant_matrix [out] Covariant matrix
     * @param is_normal_vector_negated Whether the direction of the computed
     * normal vector should be negated. By default, it is false.
     * @return Jacobian determinant for the map from the unit cell to the real
     * cell
     */
    template <int spacedim, typename RangeNumberType = double>
    HBEM_ATTR_HOST HBEM_ATTR_DEV RangeNumberType
    surface_jacobian_det_normal_vector_and_covariant(
      const RangeNumberType                  *mapping_shape_grad_matrix_data,
      const Point<spacedim, RangeNumberType> *mapping_support_points_ptr,
      const unsigned int                      mapping_n_shape_functions,
      Tensor<1, spacedim, RangeNumberType>   &normal_vector,
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
                &covariant_matrix,
      const bool is_normal_vector_negated = false)
    {
      // Currently, only @p spacedim=3 is supported.
      assert(spacedim == 3);

      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        mapping_shape_grad_matrix_at_quad_point(
          const_cast<RangeNumberType *>(mapping_shape_grad_matrix_data),
          mapping_n_shape_functions,
          2);

      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        mapping_support_point_coordinate_matrix(
          const_cast<RangeNumberType *>(
            reinterpret_cast<const RangeNumberType *>(
              mapping_support_points_ptr)),
          spacedim,
          mapping_n_shape_functions);

      // Compute the Jacobian matrix of the cell mapping.
      RangeNumberType jacobian_matrix_data[spacedim * 2];
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
        jacobian_matrix(jacobian_matrix_data, spacedim, 2);
      mapping_support_point_coordinate_matrix.mmult(
        jacobian_matrix, mapping_shape_grad_matrix_at_quad_point);

      // Compute the Gramian matrix, i.e. metric tensor.
      RangeNumberType                                             G_data[4];
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType> G(G_data,
                                                                    2,
                                                                    2);
      jacobian_matrix.Tmmult(G, jacobian_matrix);

      // Compute the covariant matrix.
      RangeNumberType                                             G_inv_data[4];
      HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType> G_inv(
        G_inv_data, 2, 2);
      G_inv.invert2x2(G);
      jacobian_matrix.mmult(covariant_matrix, G_inv);

#ifdef __CUDA_ARCH__
      RangeNumberType surface_jacobian_det = ::sqrt(G.determinant2x2());
#else
      RangeNumberType surface_jacobian_det = std::sqrt(G.determinant2x2());
#endif

      if (is_normal_vector_negated)
        {
          normal_vector[0] =
            (jacobian_matrix_data[2] * jacobian_matrix_data[4] -
             jacobian_matrix_data[1] * jacobian_matrix_data[5]) /
            surface_jacobian_det;
          normal_vector[1] =
            (jacobian_matrix_data[0] * jacobian_matrix_data[5] -
             jacobian_matrix_data[2] * jacobian_matrix_data[3]) /
            surface_jacobian_det;
          normal_vector[2] =
            (jacobian_matrix_data[1] * jacobian_matrix_data[3] -
             jacobian_matrix_data[0] * jacobian_matrix_data[4]) /
            surface_jacobian_det;
        }
      else
        {
          normal_vector[0] =
            (jacobian_matrix_data[1] * jacobian_matrix_data[5] -
             jacobian_matrix_data[2] * jacobian_matrix_data[4]) /
            surface_jacobian_det;
          normal_vector[1] =
            (jacobian_matrix_data[2] * jacobian_matrix_data[3] -
             jacobian_matrix_data[0] * jacobian_matrix_data[5]) /
            surface_jacobian_det;
          normal_vector[2] =
            (jacobian_matrix_data[0] * jacobian_matrix_data[4] -
             jacobian_matrix_data[1] * jacobian_matrix_data[3]) /
            surface_jacobian_det;
        }

      return surface_jacobian_det;
    }


    /**
     * Compute surface curl of a finite element shape function at a quadrature
     * point.
     *
     * @param fe_shape_index Index of finite element shape function
     * @param shape_grad_matrix_data Pointer to the gradient matrix data (in the
     * unit cell) of finite element shape functions at the quadrature point.
     * @param n_shape_functions Number of shape functions in the finite element
     * @param covariant_matrix Covariant matrix
     * @param normal_vector Normal vector at the quadrature point
     * @param curl [out] Surface curl of the finite element shape function at
     * the quadrature point
     */
    template <int dim, int spacedim, typename RangeNumberType = double>
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    surface_curl(
      const unsigned int     fe_shape_index,
      const RangeNumberType *shape_grad_matrix_data,
      const unsigned int     n_shape_functions,
      const HierBEM::PlatformShared::FullMatrixWrapper<RangeNumberType>
                                                 &covariant_matrix,
      const Tensor<1, spacedim, RangeNumberType> &normal_vector,
      Tensor<1, spacedim, RangeNumberType>       &curl)
    {
      RangeNumberType shape_grad_in_unit_cell_data[dim];
      RangeNumberType shape_grad_in_real_cell_data[spacedim];

      HierBEM::PlatformShared::VectorWrapper<RangeNumberType>
        shape_grad_in_unit_cell(shape_grad_in_unit_cell_data, dim);
      HierBEM::PlatformShared::VectorWrapper<RangeNumberType>
        shape_grad_in_real_cell(shape_grad_in_real_cell_data, spacedim);

      // Acquire shape function gradient in the unit cell.
      for (unsigned int d = 0; d < dim; d++)
        shape_grad_in_unit_cell(d) =
          shape_grad_matrix_data[d * n_shape_functions + fe_shape_index];

      // Push forward the gradient vector from the unit cell to the real cell.
      covariant_matrix.vmult(shape_grad_in_real_cell, shape_grad_in_unit_cell);

      // Compute surface curl.
      curl = HierBEM::PlatformShared::cross_product_3d(normal_vector,
                                                       shape_grad_in_real_cell);
    }


    /**
     * Coordinate transformation of the specified quadrature point from
     * the unit cell to the real cell based on a list of mapping support
     * points in the real cell.
     *
     * @param mapping_shape_values_ptr Pointer to the list of mapping shape
     * function values at the quadrature point.
     * @param mapping_support_points_ptr Pointer to the list of mapping support
     * points in the real cell in the lexicographic order.
     * @param mapping_n_shape_functions Number of mapping shape functions
     * @param quad_point_in_real_cell [out] Coordinates of the quadrature point
     * in the real cell
     */
    template <int spacedim, typename RangeNumberType = double>
    HBEM_ATTR_HOST HBEM_ATTR_DEV void
    transform_quad_point_from_unit_to_permuted_real_cell(
      const RangeNumberType                  *mapping_shape_values_ptr,
      const Point<spacedim, RangeNumberType> *mapping_support_points_ptr,
      const unsigned int                      mapping_n_shape_functions,
      Point<spacedim, RangeNumberType>       &quad_point_in_real_cell)
    {
      // Reset all coordinate components to zero.
      Point<spacedim, RangeNumberType> local_quad_point_in_real_cell;

#pragma unroll
      for (unsigned int i = 0; i < spacedim; i++)
        local_quad_point_in_real_cell(i) = RangeNumberType();

      // Linear combination of support point coordinates and evaluation of
      // mapping shape functions at the specified unit cell coordinates.
      for (unsigned int i = 0; i < mapping_n_shape_functions; i++)
        local_quad_point_in_real_cell =
          local_quad_point_in_real_cell +
          mapping_shape_values_ptr[i] * mapping_support_points_ptr[i];

      quad_point_in_real_cell = local_quad_point_in_real_cell;
    }
  } // namespace PlatformShared
} // namespace BEMTools

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PLATFORM_SHARED_BEM_TOOLS_H_
