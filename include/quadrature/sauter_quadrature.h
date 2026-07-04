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
 * @file sauter_quadrature.h
 * @brief Functions for computing cell values and applying quadrature for Sauter
 * quadrature.
 * @ingroup sauter_quadrature
 *
 * @date 2026-05-31
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_H_
#define HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <array>
#include <vector>

#include "bem/bem_kernels.h"
#include "bem/bem_tools.h"
#include "bem/bem_values.h"
#include "bem/cell_neighboring_type.h"
#include "config.h"
#include "mapping/mapping_info.h"
#include "platform_shared/bem_tools.h"
#include "platform_shared/full_matrix_wrapper.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Get the DoF indices associated with the cell vertices from a list of DoF
 * indices which have been arranged in either the forward or backward
 * lexicographic order. In this overloaded version, the results are returned
 * in an array as the function's return value.
 *
 * \mynote{There are <code>GeometryInfo<dim>::vertices_per_cell</code>
 * vertices in the returned array, among which the last two vertex DoF indices
 * have been swapped in this function so that the whole list of vertex DoF
 * indices in the returned array are arranged in either the clockwise or
 * counter clockwise order instead of the original lexicographic(zigzag)
 * order.}
 *
 * @param fe
 * @param dof_indices List of DoF indices in either the forward or backward
 * lexicographic order.
 * @return List of DoF indices for the cell vertices or corners with the last
 * two swapped.
 */
template <int dim, int spacedim>
std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
get_vertex_dof_indices_swapped(
  const FiniteElement<dim, spacedim>         &fe,
  const std::vector<types::global_dof_index> &dof_indices)
{
  Assert(dim == 2, ExcNotImplemented());

  std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
    vertex_dof_indices;

  /**
   * When the finite element is L2, such as @p FE_DGQ, its member
   * @p dofs_per_face is 0. Therefore, here we manually calculate the number
   * of DoFs per face.
   */
  const unsigned int dofs_per_face =
    fe.dofs_per_face > 0 ?
      fe.dofs_per_face :
      static_cast<unsigned int>(
        dealii::Utilities::fixed_power<dim - 1>(fe.degree + 1));

  vertex_dof_indices[0] = dof_indices[0];
  vertex_dof_indices[1] = dof_indices[dofs_per_face - 1];
  vertex_dof_indices[2] = dof_indices[dof_indices.size() - 1];
  vertex_dof_indices[3] =
    dof_indices[dof_indices.size() - 1 - (dofs_per_face - 1)];

  return vertex_dof_indices;
}


/**
 * Get the DoF indices associated with the cell vertices from a list of DoF
 * indices which have been arranged in either the forward or backward
 * lexicographic order. In this overloaded version, the results are returned
 * in an array as the last argument of this function.
 *
 * \mynote{There are <code>GeometryInfo<dim>::vertices_per_cell</code>
 * vertices in the returned array, among which the last two vertex DoF indices
 * have been swapped in this function so that the whole list of vertex DoF
 * indices in the returned array are arranged in either the clockwise or
 * counter clockwise order instead of the original lexicographic(zigzag)
 * order.}
 *
 * @param fe
 * @param dof_indices List of DoF indices in either the forward or backward
 * lexicographic order.
 * @param vertex_dof_indices [out] List of DoF indices for the cell vertices
 * or corners with the last two swapped.
 */
template <int dim, int spacedim>
void
get_vertex_dof_indices_swapped(
  const FiniteElement<dim, spacedim>         &fe,
  const std::vector<types::global_dof_index> &dof_indices,
  std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
    &vertex_dof_indices)
{
  Assert(dim == 2, ExcNotImplemented());

  /**
   * When the finite element is L2, such as @p FE_DGQ, its member
   * @p dofs_per_face is 0. Therefore, here we manually calculate the number
   * of DoFs per face.
   */
  const unsigned int dofs_per_face =
    fe.dofs_per_face > 0 ?
      fe.dofs_per_face :
      static_cast<unsigned int>(
        dealii::Utilities::fixed_power<dim - 1>(fe.degree + 1));

  vertex_dof_indices[0] = dof_indices[0];
  vertex_dof_indices[1] = dof_indices[dofs_per_face - 1];
  vertex_dof_indices[2] = dof_indices[dof_indices.size() - 1];
  vertex_dof_indices[3] =
    dof_indices[dof_indices.size() - 1 - (dofs_per_face - 1)];
}


/**
 * Get the local index for the starting vertex in the cell by checking
 * the list of numbers assigned to cell vertices with the last two swapped.
 *
 * \mynote{There are two cases to be processed here, common edge and common
 * vertex.
 * 1. In the common edge case, there are two DoF indices in the vector
 * <code>vertex_dof_index_intersection</code>. Then their array indices wrt.
 * the vector <code>local_vertex_dof_indices_swapped</code> will be
 * searched. By considering this vector as a closed loop list, the two DoF
 * indices in this vector are successively located and the first one of which
 * is the vertex to start subsequent DoF traversing.
 * 2. In the common vertex case, since there is only one DoF index in the
 * vector @p vertex_dof_index_intersection, this vertex is the starting point.}
 *
 * @param common_vertex_dof_indices The vector storing the pairs of vertex
 * DoF indices in \f$K_x\f$ and \f$K_y\f$, which share common vertices.
 * @param local_vertex_dof_indices_swapped Vertex DoF indices with the last
 * two swapped, which have been obtained from the function
 * @p get_vertex_dof_indices_swapped.
 * @param is_first_cell If the common vertex DoF indices in the first cell or
 * the second cell are to be extracted.
 * @return The array index for the starting vertex, wrt. the original list of
 * vertex DoF indices, i.e. the last two elements of which are not swapped.
 */
template <int vertices_per_cell>
unsigned int
get_start_vertex_local_index_in_cell_from_vertex_numbers(
  const std::vector<std::pair<unsigned int, unsigned int>>
    &common_vertex_pair_local_indices,
  const std::array<unsigned int, vertices_per_cell>
            &vertex_local_indices_in_cell_with_last_two_swapped,
  const bool is_first_cell)
{
  /**
   * The local index of the starting vertex should be in the range [0,
   * vertices_per_cell). Therefore, we use @p vertices_per_cell as its
   * initial invalid value.
   */
  unsigned int starting_vertex_local_index = vertices_per_cell;

  switch (common_vertex_pair_local_indices.size())
    {
        case 2: // Common edge case
        {
          unsigned int first_vertex_local_index;
          unsigned int second_vertex_local_index;

          if (is_first_cell)
            {
              first_vertex_local_index =
                common_vertex_pair_local_indices[0].first;
              second_vertex_local_index =
                common_vertex_pair_local_indices[1].first;
            }
          else
            {
              first_vertex_local_index =
                common_vertex_pair_local_indices[0].second;
              second_vertex_local_index =
                common_vertex_pair_local_indices[1].second;
            }

          typename std::array<unsigned int, vertices_per_cell>::const_iterator
            first_common_vertex_iterator = std::find(
              vertex_local_indices_in_cell_with_last_two_swapped.cbegin(),
              vertex_local_indices_in_cell_with_last_two_swapped.cend(),
              first_vertex_local_index);
          Assert(first_common_vertex_iterator !=
                   vertex_local_indices_in_cell_with_last_two_swapped.cend(),
                 ExcInternalError());

          if ((first_common_vertex_iterator + 1) !=
              vertex_local_indices_in_cell_with_last_two_swapped.cend())
            {
              if (*(first_common_vertex_iterator + 1) ==
                  second_vertex_local_index)
                {
                  starting_vertex_local_index = first_vertex_local_index;
                }
              else
                {
                  starting_vertex_local_index = second_vertex_local_index;
                }
            }
          else
            {
              if ((*vertex_local_indices_in_cell_with_last_two_swapped
                      .cbegin()) == second_vertex_local_index)
                {
                  starting_vertex_local_index = first_vertex_local_index;
                }
              else
                {
                  starting_vertex_local_index = second_vertex_local_index;
                }
            }

          break;
        }
        case 1: // Common vertex case
        {
          starting_vertex_local_index =
            is_first_cell ? common_vertex_pair_local_indices[0].first :
                            common_vertex_pair_local_indices[0].second;

          break;
        }
      default:
        Assert(false, ExcInternalError());
        break;
    }

  return starting_vertex_local_index;
}


/**
 * Permute mapping support points in real cells and DoF indices (using the
 * external full DoF numbering) for Sauter quadrature, the behavior of which
 * depends on the detected cell neighboring types.
 *
 * \mynote{This version involves @p PairCellWiseScratchData and
 * @p PairCellWisePerTaskData.
 *
 * @param scratch
 * @param data
 * @param cell_neighboring_type
 * @param kx_cell_iter
 * @param ky_cell_iter
 * @param kx_mapping_info
 * @param ky_mapping_info
 * @param mapping_support_point_table Mapping support points for all active
 * cells in the triangulation
 */
template <int dim, int spacedim, typename KernelNumberType>
void
permute_dofs_and_mapping_support_points_for_sauter_quad(
  PairCellWiseScratchDataBase<dim, spacedim, KernelNumberType> &scratch,
  PairCellWisePerTaskData<dim, spacedim, KernelNumberType>     &data,
  const CellNeighboringType cell_neighboring_type,
  const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
  const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
  const MappingInfo<dim, spacedim>                        &kx_mapping_info,
  const MappingInfo<dim, spacedim>                        &ky_mapping_info,
  const Table<
    2,
    Point<spacedim,
          typename numbers::NumberTraits<KernelNumberType>::real_type>>
    &mapping_support_point_table)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  // Geometry information.
  const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

  const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
  const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

  switch (cell_neighboring_type)
    {
      case CellNeighboringType::SamePanel:
        case CellNeighboringType::Regular: {
          // For the same panel and regular cell neighboring types, mapping
          // support points do not need permutation and being copied into the
          // scratch data. Only finite element DoF indices are permuted from the
          // hierarchic order to the lexicographic order.
          if (kx_fe.dofs_per_cell > 1)
            {
              BEMTools::permute_vector(
                scratch.kx_local_dof_indices_in_default_dof_order,
                scratch.kx_fe_poly_space_numbering_inverse,
                data.kx_local_dof_indices_permuted);
            }
          else
            {
              /**
               * Handle the case when the finite element order is 0, i.e. for
               * @p FE_DGQ(0). Permutation is not needed.
               */
              data.kx_local_dof_indices_permuted[0] =
                scratch.kx_local_dof_indices_in_default_dof_order[0];
            }

          if (ky_fe.dofs_per_cell > 1)
            {
              /**
               * Get DoF indices in the lexicographic
               * order.
               */
              BEMTools::permute_vector(
                scratch.ky_local_dof_indices_in_default_dof_order,
                scratch.ky_fe_poly_space_numbering_inverse,
                data.ky_local_dof_indices_permuted);
            }
          else
            {
              /**
               * Handle the case when the finite element order is 0, i.e. for
               * @p FE_DGQ. Then there is no permutation needed.
               */
              data.ky_local_dof_indices_permuted[0] =
                scratch.ky_local_dof_indices_in_default_dof_order[0];
            }

          break;
        }
        case CellNeighboringType::CommonEdge: {
          // This part handles the common edge case of Sauter's quadrature
          // rule.
          // 1. Get the DoF indices in the lexicographic order for \f$K_x\f$.
          // 2. Get the DoF indices in the reversed lexicographic order for
          // \f$K_y\f$.
          // 3. Extract only those DoF indices which are located at cell
          // vertices in \f$K_x\f$ and \f$K_y\f$. N.B. The DoF indices for the
          // last two vertices are swapped, such that the four vertices are in
          // either clockwise or counter clockwise order.
          // 4. Determine the starting vertex for \f$K_x\f$ and regenerate the
          // permutation numbering for traversing in the forward lexicographic
          // order by starting from this vertex.
          // 5. Determine the starting vertex for \f$K_y\f$ and regenerate the
          // permutation numbering for traversing in the backward
          // lexicographic order by starting from this vertex.
          // 6. Apply the newly generated permutation numbering scheme to
          // support points and DoF indices in the original default DoF order.

          // Determine the starting vertex index in \f$K_x\f$.
          unsigned int kx_starting_vertex_local_index =
            get_start_vertex_local_index_in_cell_from_vertex_numbers<
              vertices_per_cell>(scratch.common_vertex_pair_local_indices,
                                 {{0, 1, 3, 2}},
                                 true);
          AssertIndexRange(kx_starting_vertex_local_index, vertices_per_cell);

          // Determine the starting vertex index in \f$K_y\f$.
          unsigned int ky_starting_vertex_local_index =
            get_start_vertex_local_index_in_cell_from_vertex_numbers<
              vertices_per_cell>(scratch.common_vertex_pair_local_indices,
                                 {{0, 2, 3, 1}},
                                 false);
          AssertIndexRange(ky_starting_vertex_local_index, vertices_per_cell);

          // Permute mapping support points by starting from the common edge
          // in the lexicographic order.
          BEMTools::permute_vector(
            &mapping_support_point_table(kx_cell_iter->active_cell_index(), 0),
            kx_mapping_info.get_lexicographic_numberings_for_support_points()
              [kx_starting_vertex_local_index],
            scratch.kx_mapping_support_points_permuted);
          BEMTools::permute_vector(
            &mapping_support_point_table(ky_cell_iter->active_cell_index(), 0),
            ky_mapping_info
              .get_reversed_lexicographic_numberings_for_support_points()
                [ky_starting_vertex_local_index],
            scratch.ky_mapping_support_points_permuted);

          // Permute DoF indices in finite elements from the hierarchic order to
          // the lexicographic order, starting from the common edge.
          if (kx_fe.dofs_per_cell > 1)
            {
              // Generate the permutation of DoFs in \f$K_x\f$ by starting
              // from <code>kx_starting_vertex_index</code>.
              BEMTools::generate_forward_dof_permutation(
                kx_fe,
                kx_starting_vertex_local_index,
                scratch.kx_local_dof_permutation);

              BEMTools::permute_vector(
                scratch.kx_local_dof_indices_in_default_dof_order,
                scratch.kx_local_dof_permutation,
                data.kx_local_dof_indices_permuted);
            }
          else
            {
              /**
               * Handle the case when the finite element order is 0, i.e. for
               * @p FE_DGQ. Then there is no permutation needed.
               */
              data.kx_local_dof_indices_permuted[0] =
                scratch.kx_local_dof_indices_in_default_dof_order[0];
            }

          if (ky_fe.dofs_per_cell > 1)
            {
              // Generate the permutation of DoFs in \f$K_y\f$ by starting
              // from <code>ky_starting_vertex_index</code>.
              BEMTools::generate_backward_dof_permutation(
                ky_fe,
                ky_starting_vertex_local_index,
                scratch.ky_local_dof_permutation);

              BEMTools::permute_vector(
                scratch.ky_local_dof_indices_in_default_dof_order,
                scratch.ky_local_dof_permutation,
                data.ky_local_dof_indices_permuted);
            }
          else
            {
              /**
               * Handle the case when the finite element order is 0, i.e. for
               * @p FE_DGQ. Then there is no permutation needed.
               */
              data.ky_local_dof_indices_permuted[0] =
                scratch.ky_local_dof_indices_in_default_dof_order[0];
            }

          break;
        }
        case CellNeighboringType::CommonVertex: {
          // Determine the starting vertex index in \f$K_x\f$.
          unsigned int kx_starting_vertex_local_index =
            get_start_vertex_local_index_in_cell_from_vertex_numbers<
              vertices_per_cell>(scratch.common_vertex_pair_local_indices,
                                 {{0, 1, 3, 2}},
                                 true);
          AssertIndexRange(kx_starting_vertex_local_index, vertices_per_cell);

          // Determine the starting vertex index in \f$K_y\f$.
          unsigned int ky_starting_vertex_local_index =
            get_start_vertex_local_index_in_cell_from_vertex_numbers<
              vertices_per_cell>(scratch.common_vertex_pair_local_indices,
                                 {{0, 2, 3, 1}},
                                 false);
          AssertIndexRange(ky_starting_vertex_local_index, vertices_per_cell);

          // Permute mapping support points by starting from the common vertex
          // in the lexicographic order.
          BEMTools::permute_vector(
            &mapping_support_point_table(kx_cell_iter->active_cell_index(), 0),
            kx_mapping_info.get_lexicographic_numberings_for_support_points()
              [kx_starting_vertex_local_index],
            scratch.kx_mapping_support_points_permuted);
          BEMTools::permute_vector(
            &mapping_support_point_table(ky_cell_iter->active_cell_index(), 0),
            ky_mapping_info.get_lexicographic_numberings_for_support_points()
              [ky_starting_vertex_local_index],
            scratch.ky_mapping_support_points_permuted);

          // Permute DoF indices in finite elements from the hierarchic order to
          // the lexicographic order, starting from the common vertex.
          if (kx_fe.dofs_per_cell > 1)
            {
              // Generate the permutation of DoFs in \f$K_x\f$ by starting
              // from <code>kx_starting_vertex_index</code>.
              BEMTools::generate_forward_dof_permutation(
                kx_fe,
                kx_starting_vertex_local_index,
                scratch.kx_local_dof_permutation);

              BEMTools::permute_vector(
                scratch.kx_local_dof_indices_in_default_dof_order,
                scratch.kx_local_dof_permutation,
                data.kx_local_dof_indices_permuted);
            }
          else
            {
              /**
               * Handle the case when the finite element order is 0, i.e. for
               * @p FE_DGQ. Then there is no permutation needed.
               */
              data.kx_local_dof_indices_permuted[0] =
                scratch.kx_local_dof_indices_in_default_dof_order[0];
            }

          if (ky_fe.dofs_per_cell > 1)
            {
              // Generate the permutation of DoFs in \f$K_y\f$ by starting
              // from <code>ky_starting_vertex_index</code>.
              BEMTools::generate_forward_dof_permutation(
                ky_fe,
                ky_starting_vertex_local_index,
                scratch.ky_local_dof_permutation);

              BEMTools::permute_vector(
                scratch.ky_local_dof_indices_in_default_dof_order,
                scratch.ky_local_dof_permutation,
                data.ky_local_dof_indices_permuted);
            }
          else
            {
              /**
               * Handle the case when the finite element order is 0, i.e. for
               * @p FE_DGQ. Then there is no permutation needed.
               */
              data.ky_local_dof_indices_permuted[0] =
                scratch.ky_local_dof_indices_in_default_dof_order[0];
            }

          break;
        }
        default: {
          Assert(false, ExcInternalError());
          break;
        }
    }
}


/**
 * Compute cell values at quadrature points (surface Jacobians, normal
 * vectors, quadrature points and surface curls of all finite element shape
 * functions) in the real cell to be used in the Sauter quadrature for a pair
 * of cells.
 *
 * This function is only used for the same panel cell neighboring type in the
 * full matrix assembler and only supports building matrices on the whole
 * triangulation.
 *
 * @param kx_cell_index_global Global index of the active cell Kx
 * @param ky_cell_index_global Global index of the active cell Ky
 * @param kx_mapping_index
 * @param ky_mapping_index
 * @param kx_mapping_n_shape_functions
 * @param ky_mapping_n_shape_functions
 * @param is_kx_normal_inward
 * @param is_ky_normal_inward
 * @param bem_values
 * @param scratch_data
 */
template <int dim, int spacedim, typename KernelNumberType>
void
compute_cell_values_for_all_shape_functions_same_panel(
  const types::global_cell_index kx_cell_index_global,
  const types::global_cell_index ky_cell_index_global,
  const unsigned int             kx_mapping_index,
  const unsigned int             ky_mapping_index,
  const unsigned int             kx_mapping_n_shape_functions,
  const unsigned int             ky_mapping_n_shape_functions,
  const bool                     is_kx_normal_inward,
  const bool                     is_ky_normal_inward,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
    &scratch_data)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(kx_cell_index_global == ky_cell_index_global, ExcInternalError());
  Assert(kx_mapping_index == ky_mapping_index, ExcInternalError());
  Assert(kx_mapping_n_shape_functions == ky_mapping_n_shape_functions,
         ExcInternalError());
  Assert(is_kx_normal_inward == is_ky_normal_inward, ExcInternalError());
  Assert(bem_values.is_surface_curl_needed ==
           scratch_data.is_surface_curl_needed,
         ExcInternalError());
  Assert(scratch_data.common_vertex_pair_local_indices.size() ==
           GeometryInfo<dim>::vertices_per_cell,
         ExcInternalError());

  for (unsigned int k3_index = 0; k3_index < 8; k3_index++)
    {
      for (unsigned int quad_no = 0;
           quad_no < bem_values.quad_rule_for_same_panel.size();
           quad_no++)
        {
          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.kx_mapping_shape_value_table_for_same_panel(
                k3_index, quad_no, kx_mapping_index, 0),
              &bem_values.mapping_support_point_table(kx_cell_index_global, 0),
              kx_mapping_n_shape_functions,
              scratch_data.kx_quad_points_same_panel(k3_index, quad_no));

          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.ky_mapping_shape_value_table_for_same_panel(
                k3_index, quad_no, ky_mapping_index, 0),
              &bem_values.mapping_support_point_table(ky_cell_index_global, 0),
              ky_mapping_n_shape_functions,
              scratch_data.ky_quad_points_same_panel(k3_index, quad_no));

          if (bem_values.is_surface_curl_needed)
            {
              // Covariant matrix is computed on the stack without storing
              // into data tables.
              real_type kx_covariant_matrix_data[spacedim * dim];
              real_type ky_covariant_matrix_data[spacedim * dim];
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                kx_covariant_matrix(kx_covariant_matrix_data, spacedim, dim);
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                ky_covariant_matrix(ky_covariant_matrix_data, spacedim, dim);

              scratch_data.kx_jacobians_same_panel(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_same_panel(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    &bem_values.mapping_support_point_table(
                      kx_cell_index_global, 0),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_same_panel(k3_index, quad_no),
                    kx_covariant_matrix,
                    is_kx_normal_inward);

              scratch_data.ky_jacobians_same_panel(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_same_panel(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    &bem_values.mapping_support_point_table(
                      ky_cell_index_global, 0),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_same_panel(k3_index, quad_no),
                    ky_covariant_matrix,
                    is_ky_normal_inward);

              // Iterate over each shape function in the finite element for
              // Kx.
              for (unsigned int fe_shape_index = 0;
                   fe_shape_index < bem_values.kx_fe.dofs_per_cell;
                   fe_shape_index++)
                {
                  BEMTools::PlatformShared::
                    surface_curl<dim, spacedim, real_type>(
                      fe_shape_index,
                      &bem_values
                         .kx_shape_grad_matrix_table_for_same_panel(k3_index,
                                                                    quad_no)
                         .begin()
                         ->value(),
                      bem_values.kx_fe.dofs_per_cell,
                      kx_covariant_matrix,
                      scratch_data.kx_normals_same_panel(k3_index, quad_no),
                      scratch_data.kx_shape_curls_same_panel(fe_shape_index,
                                                             k3_index,
                                                             quad_no));
                }

              // Iterate over each shape function in the finite element for
              // Ky.
              for (unsigned int fe_shape_index = 0;
                   fe_shape_index < bem_values.ky_fe.dofs_per_cell;
                   fe_shape_index++)
                {
                  BEMTools::PlatformShared::
                    surface_curl<dim, spacedim, real_type>(
                      fe_shape_index,
                      &bem_values
                         .ky_shape_grad_matrix_table_for_same_panel(k3_index,
                                                                    quad_no)
                         .begin()
                         ->value(),
                      bem_values.ky_fe.dofs_per_cell,
                      ky_covariant_matrix,
                      scratch_data.ky_normals_same_panel(k3_index, quad_no),
                      scratch_data.ky_shape_curls_same_panel(fe_shape_index,
                                                             k3_index,
                                                             quad_no));
                }
            }
          else
            {
              scratch_data.kx_jacobians_same_panel(k3_index,
                                                   quad_no) = BEMTools::
                PlatformShared::surface_jacobian_det_and_normal_vector(
                  &bem_values
                     .kx_mapping_shape_grad_matrix_table_for_same_panel(
                       kx_mapping_index, k3_index, quad_no)
                     .begin()
                     ->value(),
                  &bem_values.mapping_support_point_table(kx_cell_index_global,
                                                          0),
                  kx_mapping_n_shape_functions,
                  scratch_data.kx_normals_same_panel(k3_index, quad_no),
                  is_kx_normal_inward);

              scratch_data.ky_jacobians_same_panel(k3_index,
                                                   quad_no) = BEMTools::
                PlatformShared::surface_jacobian_det_and_normal_vector(
                  &bem_values
                     .ky_mapping_shape_grad_matrix_table_for_same_panel(
                       ky_mapping_index, k3_index, quad_no)
                     .begin()
                     ->value(),
                  &bem_values.mapping_support_point_table(ky_cell_index_global,
                                                          0),
                  ky_mapping_n_shape_functions,
                  scratch_data.ky_normals_same_panel(k3_index, quad_no),
                  is_ky_normal_inward);
            }
        }
    }
}


/**
 * Compute cell values at quadrature points (surface Jacobians, normal
 * vectors, quadrature points and surface curls of all finite element shape
 * functions) in the real cell to be used in the Sauter quadrature for a pair
 * of cells.
 *
 * This function is only used for the common edge cell neighboring type in the
 * full matrix assembler and only supports building matrices on the whole
 * triangulation.
 *
 * @param kx_mapping_index
 * @param ky_mapping_index
 * @param kx_mapping_n_shape_functions
 * @param ky_mapping_n_shape_functions
 * @param is_kx_normal_inward
 * @param is_ky_normal_inward
 * @param bem_values
 * @param scratch_data
 */
template <int dim, int spacedim, typename KernelNumberType>
void
compute_cell_values_for_all_shape_functions_common_edge(
  const unsigned int kx_mapping_index,
  const unsigned int ky_mapping_index,
  const unsigned int kx_mapping_n_shape_functions,
  const unsigned int ky_mapping_n_shape_functions,
  const bool         is_kx_normal_inward,
  const bool         is_ky_normal_inward,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
    &scratch_data)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(bem_values.is_surface_curl_needed ==
           scratch_data.is_surface_curl_needed,
         ExcInternalError());
  Assert(scratch_data.common_vertex_pair_local_indices.size() ==
           GeometryInfo<2>::vertices_per_face,
         ExcInternalError());

  for (unsigned int k3_index = 0; k3_index < 6; k3_index++)
    {
      for (unsigned int quad_no = 0;
           quad_no < bem_values.quad_rule_for_common_edge.size();
           quad_no++)
        {
          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.kx_mapping_shape_value_table_for_common_edge(
                k3_index, quad_no, kx_mapping_index, 0),
              scratch_data.kx_mapping_support_points_permuted.data(),
              kx_mapping_n_shape_functions,
              scratch_data.kx_quad_points_common_edge(k3_index, quad_no));

          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.ky_mapping_shape_value_table_for_common_edge(
                k3_index, quad_no, ky_mapping_index, 0),
              scratch_data.ky_mapping_support_points_permuted.data(),
              ky_mapping_n_shape_functions,
              scratch_data.ky_quad_points_common_edge(k3_index, quad_no));

          if (bem_values.is_surface_curl_needed)
            {
              // Covariant matrix is computed on the stack without storing
              // into data tables.
              real_type kx_covariant_matrix_data[spacedim * dim];
              real_type ky_covariant_matrix_data[spacedim * dim];
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                kx_covariant_matrix(kx_covariant_matrix_data, spacedim, dim);
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                ky_covariant_matrix(ky_covariant_matrix_data, spacedim, dim);

              scratch_data.kx_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_edge(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_edge(k3_index, quad_no),
                    kx_covariant_matrix,
                    is_kx_normal_inward);

              // Because the mapping support points in ky have been reversed,
              // the flag @p is_ky_normal_inward should be negated.
              scratch_data.ky_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_edge(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_edge(k3_index, quad_no),
                    ky_covariant_matrix,
                    !is_ky_normal_inward);

              // Iterate over each shape function in the finite element for
              // Kx.
              for (unsigned int fe_shape_index = 0;
                   fe_shape_index < bem_values.kx_fe.dofs_per_cell;
                   fe_shape_index++)
                {
                  BEMTools::PlatformShared::
                    surface_curl<dim, spacedim, real_type>(
                      fe_shape_index,
                      &bem_values
                         .kx_shape_grad_matrix_table_for_common_edge(k3_index,
                                                                     quad_no)
                         .begin()
                         ->value(),
                      bem_values.kx_fe.dofs_per_cell,
                      kx_covariant_matrix,
                      scratch_data.kx_normals_common_edge(k3_index, quad_no),
                      scratch_data.kx_shape_curls_common_edge(fe_shape_index,
                                                              k3_index,
                                                              quad_no));
                }

              // Iterate over each shape function in the finite element for
              // Ky.
              for (unsigned int fe_shape_index = 0;
                   fe_shape_index < bem_values.ky_fe.dofs_per_cell;
                   fe_shape_index++)
                {
                  BEMTools::PlatformShared::
                    surface_curl<dim, spacedim, real_type>(
                      fe_shape_index,
                      &bem_values
                         .ky_shape_grad_matrix_table_for_common_edge(k3_index,
                                                                     quad_no)
                         .begin()
                         ->value(),
                      bem_values.ky_fe.dofs_per_cell,
                      ky_covariant_matrix,
                      scratch_data.ky_normals_common_edge(k3_index, quad_no),
                      scratch_data.ky_shape_curls_common_edge(fe_shape_index,
                                                              k3_index,
                                                              quad_no));
                }
            }
          else
            {
              scratch_data.kx_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_edge(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_edge(k3_index, quad_no),
                    is_kx_normal_inward);

              // Because the mapping support points in ky have been reversed,
              // the flag @p is_ky_normal_inward should be negated.
              scratch_data.ky_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_edge(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_edge(k3_index, quad_no),
                    !is_ky_normal_inward);
            }
        }
    }
}


/**
 * Compute cell values at quadrature points (surface Jacobians, normal
 * vectors, quadrature points and surface curls of all finite element shape
 * functions) in the real cell to be used in the Sauter quadrature for a pair
 * of cells.
 *
 * This function is only used for the common vertex cell neighboring type in
 * the full matrix assembler and only supports building matrices on the whole
 * triangulation.
 *
 * @param kx_mapping_index
 * @param ky_mapping_index
 * @param kx_mapping_n_shape_functions
 * @param ky_mapping_n_shape_functions
 * @param is_kx_normal_inward
 * @param is_ky_normal_inward
 * @param bem_values
 * @param scratch_data
 */
template <int dim, int spacedim, typename KernelNumberType>
void
compute_cell_values_for_all_shape_functions_common_vertex(
  const unsigned int kx_mapping_index,
  const unsigned int ky_mapping_index,
  const unsigned int kx_mapping_n_shape_functions,
  const unsigned int ky_mapping_n_shape_functions,
  const bool         is_kx_normal_inward,
  const bool         is_ky_normal_inward,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
    &scratch_data)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(bem_values.is_surface_curl_needed ==
           scratch_data.is_surface_curl_needed,
         ExcInternalError());
  Assert(scratch_data.common_vertex_pair_local_indices.size() == 1,
         ExcInternalError());

  for (unsigned int k3_index = 0; k3_index < 4; k3_index++)
    {
      for (unsigned int quad_no = 0;
           quad_no < bem_values.quad_rule_for_common_vertex.size();
           quad_no++)
        {
          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.kx_mapping_shape_value_table_for_common_vertex(
                k3_index, quad_no, kx_mapping_index, 0),
              scratch_data.kx_mapping_support_points_permuted.data(),
              kx_mapping_n_shape_functions,
              scratch_data.kx_quad_points_common_vertex(k3_index, quad_no));

          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.ky_mapping_shape_value_table_for_common_vertex(
                k3_index, quad_no, ky_mapping_index, 0),
              scratch_data.ky_mapping_support_points_permuted.data(),
              ky_mapping_n_shape_functions,
              scratch_data.ky_quad_points_common_vertex(k3_index, quad_no));

          if (bem_values.is_surface_curl_needed)
            {
              // Covariant matrix is computed on the stack without storing
              // into data tables.
              real_type kx_covariant_matrix_data[spacedim * dim];
              real_type ky_covariant_matrix_data[spacedim * dim];
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                kx_covariant_matrix(kx_covariant_matrix_data, spacedim, dim);
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                ky_covariant_matrix(ky_covariant_matrix_data, spacedim, dim);

              scratch_data.kx_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_vertex(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_vertex(k3_index, quad_no),
                    kx_covariant_matrix,
                    is_kx_normal_inward);

              scratch_data.ky_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_vertex(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_vertex(k3_index, quad_no),
                    ky_covariant_matrix,
                    is_ky_normal_inward);

              // Iterate over each shape function in the finite element for
              // Kx.
              for (unsigned int fe_shape_index = 0;
                   fe_shape_index < bem_values.kx_fe.dofs_per_cell;
                   fe_shape_index++)
                {
                  BEMTools::PlatformShared::
                    surface_curl<dim, spacedim, real_type>(
                      fe_shape_index,
                      &bem_values
                         .kx_shape_grad_matrix_table_for_common_vertex(k3_index,
                                                                       quad_no)
                         .begin()
                         ->value(),
                      bem_values.kx_fe.dofs_per_cell,
                      kx_covariant_matrix,
                      scratch_data.kx_normals_common_vertex(k3_index, quad_no),
                      scratch_data.kx_shape_curls_common_vertex(fe_shape_index,
                                                                k3_index,
                                                                quad_no));
                }

              // Iterate over each shape function in the finite element for
              // Ky.
              for (unsigned int fe_shape_index = 0;
                   fe_shape_index < bem_values.ky_fe.dofs_per_cell;
                   fe_shape_index++)
                {
                  BEMTools::PlatformShared::
                    surface_curl<dim, spacedim, real_type>(
                      fe_shape_index,
                      &bem_values
                         .ky_shape_grad_matrix_table_for_common_vertex(k3_index,
                                                                       quad_no)
                         .begin()
                         ->value(),
                      bem_values.ky_fe.dofs_per_cell,
                      ky_covariant_matrix,
                      scratch_data.ky_normals_common_vertex(k3_index, quad_no),
                      scratch_data.ky_shape_curls_common_vertex(fe_shape_index,
                                                                k3_index,
                                                                quad_no));
                }
            }
          else
            {
              scratch_data.kx_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_vertex(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_vertex(k3_index, quad_no),
                    is_kx_normal_inward);

              scratch_data.ky_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_vertex(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_vertex(k3_index, quad_no),
                    is_ky_normal_inward);
            }
        }
    }
}


/**
 * Compute cell values at quadrature points (surface Jacobians, normal
 * vectors, quadrature points and surface curls of a pair of finite element
 * shape functions) in the real cell to be used in the Sauter quadrature for a
 * pair of cells.
 *
 * This function is only used for the same panel cell neighboring type in the
 * full matrix assembler and only supports building matrices on the whole
 * triangulation.
 *
 * @param kx_dof_index_local
 * @param ky_dof_index_local
 * @param kx_cell_index_global Global index of the active cell Kx
 * @param ky_cell_index_global Global index of the active cell Ky
 * @param kx_mapping_index
 * @param ky_mapping_index
 * @param kx_mapping_n_shape_functions
 * @param ky_mapping_n_shape_functions
 * @param is_kx_normal_inward
 * @param is_ky_normal_inward
 * @param bem_values
 * @param scratch_data
 */
template <int dim, int spacedim, typename KernelNumberType>
void
compute_cell_values_for_one_pair_of_shape_functions_same_panel(
  const unsigned int             kx_dof_index_local,
  const unsigned int             ky_dof_index_local,
  const types::global_cell_index kx_cell_index_global,
  const types::global_cell_index ky_cell_index_global,
  const unsigned int             kx_mapping_index,
  const unsigned int             ky_mapping_index,
  const unsigned int             kx_mapping_n_shape_functions,
  const unsigned int             ky_mapping_n_shape_functions,
  const bool                     is_kx_normal_inward,
  const bool                     is_ky_normal_inward,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
    &scratch_data)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(bem_values.is_surface_curl_needed ==
           scratch_data.is_surface_curl_needed,
         ExcInternalError());
  Assert(scratch_data.common_vertex_pair_local_indices.size() ==
           GeometryInfo<dim>::vertices_per_cell,
         ExcInternalError());

  for (unsigned int k3_index = 0; k3_index < 8; k3_index++)
    {
      for (unsigned int quad_no = 0;
           quad_no < bem_values.quad_rule_for_same_panel.size();
           quad_no++)
        {
          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.kx_mapping_shape_value_table_for_same_panel(
                k3_index, quad_no, kx_mapping_index, 0),
              &bem_values.mapping_support_point_table(kx_cell_index_global, 0),
              kx_mapping_n_shape_functions,
              scratch_data.kx_quad_points_same_panel(k3_index, quad_no));

          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.ky_mapping_shape_value_table_for_same_panel(
                k3_index, quad_no, ky_mapping_index, 0),
              &bem_values.mapping_support_point_table(ky_cell_index_global, 0),
              ky_mapping_n_shape_functions,
              scratch_data.ky_quad_points_same_panel(k3_index, quad_no));

          if (bem_values.is_surface_curl_needed)
            {
              // Covariant matrix is computed on the stack without storing
              // into data tables.
              real_type kx_covariant_matrix_data[spacedim * dim];
              real_type ky_covariant_matrix_data[spacedim * dim];
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                kx_covariant_matrix(kx_covariant_matrix_data, spacedim, dim);
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                ky_covariant_matrix(ky_covariant_matrix_data, spacedim, dim);

              scratch_data.kx_jacobians_same_panel(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_same_panel(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    &bem_values.mapping_support_point_table(
                      kx_cell_index_global, 0),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_same_panel(k3_index, quad_no),
                    kx_covariant_matrix,
                    is_kx_normal_inward);

              scratch_data.ky_jacobians_same_panel(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_same_panel(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    &bem_values.mapping_support_point_table(
                      ky_cell_index_global, 0),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_same_panel(k3_index, quad_no),
                    ky_covariant_matrix,
                    is_ky_normal_inward);

              BEMTools::PlatformShared::surface_curl<dim, spacedim, real_type>(
                kx_dof_index_local,
                &bem_values
                   .kx_shape_grad_matrix_table_for_same_panel(k3_index, quad_no)
                   .begin()
                   ->value(),
                bem_values.kx_fe.dofs_per_cell,
                kx_covariant_matrix,
                scratch_data.kx_normals_same_panel(k3_index, quad_no),
                scratch_data.kx_shape_curls_same_panel(kx_dof_index_local,
                                                       k3_index,
                                                       quad_no));

              BEMTools::PlatformShared::surface_curl<dim, spacedim, real_type>(
                ky_dof_index_local,
                &bem_values
                   .ky_shape_grad_matrix_table_for_same_panel(k3_index, quad_no)
                   .begin()
                   ->value(),
                bem_values.ky_fe.dofs_per_cell,
                ky_covariant_matrix,
                scratch_data.ky_normals_same_panel(k3_index, quad_no),
                scratch_data.ky_shape_curls_same_panel(ky_dof_index_local,
                                                       k3_index,
                                                       quad_no));
            }
          else
            {
              scratch_data.kx_jacobians_same_panel(k3_index,
                                                   quad_no) = BEMTools::
                PlatformShared::surface_jacobian_det_and_normal_vector(
                  &bem_values
                     .kx_mapping_shape_grad_matrix_table_for_same_panel(
                       kx_mapping_index, k3_index, quad_no)
                     .begin()
                     ->value(),
                  &bem_values.mapping_support_point_table(kx_cell_index_global,
                                                          0),
                  kx_mapping_n_shape_functions,
                  scratch_data.kx_normals_same_panel(k3_index, quad_no),
                  is_kx_normal_inward);

              scratch_data.ky_jacobians_same_panel(k3_index,
                                                   quad_no) = BEMTools::
                PlatformShared::surface_jacobian_det_and_normal_vector(
                  &bem_values
                     .ky_mapping_shape_grad_matrix_table_for_same_panel(
                       ky_mapping_index, k3_index, quad_no)
                     .begin()
                     ->value(),
                  &bem_values.mapping_support_point_table(ky_cell_index_global,
                                                          0),
                  ky_mapping_n_shape_functions,
                  scratch_data.ky_normals_same_panel(k3_index, quad_no),
                  is_ky_normal_inward);
            }
        }
    }
}


/**
 * Compute cell values at quadrature points (surface Jacobians, normal
 * vectors, quadrature points and surface curls of a pair of finite element
 * shape functions) in the real cell to be used in the Sauter quadrature for a
 * pair of cells.
 *
 * This function is only used for the common edge cell neighboring type in the
 * full matrix assembler and only supports building matrices on the whole
 * triangulation.
 *
 * @param kx_dof_index_local
 * @param ky_dof_index_local
 * @param kx_mapping_index
 * @param ky_mapping_index
 * @param kx_mapping_n_shape_functions
 * @param ky_mapping_n_shape_functions
 * @param is_kx_normal_inward
 * @param is_ky_normal_inward
 * @param bem_values
 * @param scratch_data
 */
template <int dim, int spacedim, typename KernelNumberType>
void
compute_cell_values_for_one_pair_of_shape_functions_common_edge(
  const unsigned int kx_dof_index_local,
  const unsigned int ky_dof_index_local,
  const unsigned int kx_mapping_index,
  const unsigned int ky_mapping_index,
  const unsigned int kx_mapping_n_shape_functions,
  const unsigned int ky_mapping_n_shape_functions,
  const bool         is_kx_normal_inward,
  const bool         is_ky_normal_inward,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
    &scratch_data)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(bem_values.is_surface_curl_needed ==
           scratch_data.is_surface_curl_needed,
         ExcInternalError());
  Assert(scratch_data.common_vertex_pair_local_indices.size() ==
           GeometryInfo<2>::vertices_per_face,
         ExcInternalError());

  for (unsigned int k3_index = 0; k3_index < 6; k3_index++)
    {
      for (unsigned int quad_no = 0;
           quad_no < bem_values.quad_rule_for_common_edge.size();
           quad_no++)
        {
          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.kx_mapping_shape_value_table_for_common_edge(
                k3_index, quad_no, kx_mapping_index, 0),
              scratch_data.kx_mapping_support_points_permuted.data(),
              kx_mapping_n_shape_functions,
              scratch_data.kx_quad_points_common_edge(k3_index, quad_no));

          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.ky_mapping_shape_value_table_for_common_edge(
                k3_index, quad_no, ky_mapping_index, 0),
              scratch_data.ky_mapping_support_points_permuted.data(),
              ky_mapping_n_shape_functions,
              scratch_data.ky_quad_points_common_edge(k3_index, quad_no));

          if (bem_values.is_surface_curl_needed)
            {
              // Covariant matrix is computed on the stack without storing
              // into data tables.
              real_type kx_covariant_matrix_data[spacedim * dim];
              real_type ky_covariant_matrix_data[spacedim * dim];
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                kx_covariant_matrix(kx_covariant_matrix_data, spacedim, dim);
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                ky_covariant_matrix(ky_covariant_matrix_data, spacedim, dim);

              scratch_data.kx_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_edge(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_edge(k3_index, quad_no),
                    kx_covariant_matrix,
                    is_kx_normal_inward);

              // Because the mapping support points in ky have been reversed,
              // the flag @p is_ky_normal_inward should be negated.
              scratch_data.ky_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_edge(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_edge(k3_index, quad_no),
                    ky_covariant_matrix,
                    !is_ky_normal_inward);

              BEMTools::PlatformShared::surface_curl<dim, spacedim, real_type>(
                kx_dof_index_local,
                &bem_values
                   .kx_shape_grad_matrix_table_for_common_edge(k3_index,
                                                               quad_no)
                   .begin()
                   ->value(),
                bem_values.kx_fe.dofs_per_cell,
                kx_covariant_matrix,
                scratch_data.kx_normals_common_edge(k3_index, quad_no),
                scratch_data.kx_shape_curls_common_edge(kx_dof_index_local,
                                                        k3_index,
                                                        quad_no));

              BEMTools::PlatformShared::surface_curl<dim, spacedim, real_type>(
                ky_dof_index_local,
                &bem_values
                   .ky_shape_grad_matrix_table_for_common_edge(k3_index,
                                                               quad_no)
                   .begin()
                   ->value(),
                bem_values.ky_fe.dofs_per_cell,
                ky_covariant_matrix,
                scratch_data.ky_normals_common_edge(k3_index, quad_no),
                scratch_data.ky_shape_curls_common_edge(ky_dof_index_local,
                                                        k3_index,
                                                        quad_no));
            }
          else
            {
              scratch_data.kx_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_edge(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_edge(k3_index, quad_no),
                    is_kx_normal_inward);

              // Because the mapping support points in ky have been reversed,
              // the flag @p is_ky_normal_inward should be negated.
              scratch_data.ky_jacobians_common_edge(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_edge(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_edge(k3_index, quad_no),
                    !is_ky_normal_inward);
            }
        }
    }
}


/**
 * Compute cell values at quadrature points (surface Jacobians, normal
 * vectors, quadrature points and surface curls of a pair of finite element
 * shape functions) in the real cell to be used in the Sauter quadrature for a
 * pair of cells.
 *
 * This function is only used for the common vertex cell neighboring type in
 * the full matrix assembler and only supports building matrices on the whole
 * triangulation.
 *
 * @param kx_dof_index_local
 * @param ky_dof_index_local
 * @param kx_mapping_index
 * @param ky_mapping_index
 * @param kx_mapping_n_shape_functions
 * @param ky_mapping_n_shape_functions
 * @param is_kx_normal_inward
 * @param is_ky_normal_inward
 * @param bem_values
 * @param scratch_data
 */
template <int dim, int spacedim, typename KernelNumberType>
void
compute_cell_values_for_one_pair_of_shape_functions_common_vertex(
  const unsigned int kx_dof_index_local,
  const unsigned int ky_dof_index_local,
  const unsigned int kx_mapping_index,
  const unsigned int ky_mapping_index,
  const unsigned int kx_mapping_n_shape_functions,
  const unsigned int ky_mapping_n_shape_functions,
  const bool         is_kx_normal_inward,
  const bool         is_ky_normal_inward,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
    &scratch_data)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(bem_values.is_surface_curl_needed ==
           scratch_data.is_surface_curl_needed,
         ExcInternalError());
  Assert(scratch_data.common_vertex_pair_local_indices.size() == 1,
         ExcInternalError());

  for (unsigned int k3_index = 0; k3_index < 4; k3_index++)
    {
      for (unsigned int quad_no = 0;
           quad_no < bem_values.quad_rule_for_common_vertex.size();
           quad_no++)
        {
          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.kx_mapping_shape_value_table_for_common_vertex(
                k3_index, quad_no, kx_mapping_index, 0),
              scratch_data.kx_mapping_support_points_permuted.data(),
              kx_mapping_n_shape_functions,
              scratch_data.kx_quad_points_common_vertex(k3_index, quad_no));

          BEMTools::PlatformShared::
            transform_quad_point_from_unit_to_permuted_real_cell(
              &bem_values.ky_mapping_shape_value_table_for_common_vertex(
                k3_index, quad_no, ky_mapping_index, 0),
              scratch_data.ky_mapping_support_points_permuted.data(),
              ky_mapping_n_shape_functions,
              scratch_data.ky_quad_points_common_vertex(k3_index, quad_no));

          if (bem_values.is_surface_curl_needed)
            {
              // Covariant matrix is computed on the stack without storing
              // into data tables.
              real_type kx_covariant_matrix_data[spacedim * dim];
              real_type ky_covariant_matrix_data[spacedim * dim];
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                kx_covariant_matrix(kx_covariant_matrix_data, spacedim, dim);
              HierBEM::PlatformShared::FullMatrixWrapper<real_type>
                ky_covariant_matrix(ky_covariant_matrix_data, spacedim, dim);

              scratch_data.kx_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_vertex(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_vertex(k3_index, quad_no),
                    kx_covariant_matrix,
                    is_kx_normal_inward);

              scratch_data.ky_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_normal_vector_and_covariant(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_vertex(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_vertex(k3_index, quad_no),
                    ky_covariant_matrix,
                    is_ky_normal_inward);

              BEMTools::PlatformShared::surface_curl<dim, spacedim, real_type>(
                kx_dof_index_local,
                &bem_values
                   .kx_shape_grad_matrix_table_for_common_vertex(k3_index,
                                                                 quad_no)
                   .begin()
                   ->value(),
                bem_values.kx_fe.dofs_per_cell,
                kx_covariant_matrix,
                scratch_data.kx_normals_common_vertex(k3_index, quad_no),
                scratch_data.kx_shape_curls_common_vertex(kx_dof_index_local,
                                                          k3_index,
                                                          quad_no));

              BEMTools::PlatformShared::surface_curl<dim, spacedim, real_type>(
                ky_dof_index_local,
                &bem_values
                   .ky_shape_grad_matrix_table_for_common_vertex(k3_index,
                                                                 quad_no)
                   .begin()
                   ->value(),
                bem_values.ky_fe.dofs_per_cell,
                ky_covariant_matrix,
                scratch_data.ky_normals_common_vertex(k3_index, quad_no),
                scratch_data.ky_shape_curls_common_vertex(ky_dof_index_local,
                                                          k3_index,
                                                          quad_no));
            }
          else
            {
              scratch_data.kx_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .kx_mapping_shape_grad_matrix_table_for_common_vertex(
                         kx_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.kx_mapping_support_points_permuted.data(),
                    kx_mapping_n_shape_functions,
                    scratch_data.kx_normals_common_vertex(k3_index, quad_no),
                    is_kx_normal_inward);

              scratch_data.ky_jacobians_common_vertex(k3_index, quad_no) =
                BEMTools::PlatformShared::
                  surface_jacobian_det_and_normal_vector(
                    &bem_values
                       .ky_mapping_shape_grad_matrix_table_for_common_vertex(
                         ky_mapping_index, k3_index, quad_no)
                       .begin()
                       ->value(),
                    scratch_data.ky_mapping_support_points_permuted.data(),
                    ky_mapping_n_shape_functions,
                    scratch_data.ky_normals_common_vertex(k3_index, quad_no),
                    is_ky_normal_inward);
            }
        }
    }
}


/**
 * Apply the Sauter's quadrature rule to the kernel function pulled back to
 * the Sauter's parametric space. The result will also be multiplied by a
 * factor.
 *
 * This version is used for same panel, common edge and common vertex cell
 * neighboring types.
 *
 * @param f
 * @param factor
 * @param bem_values
 * @param scratch_data
 * @param component
 * @return
 */
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelNumberType
ApplyQuadratureUsingBEMValues(
  const KernelPulledbackToSauterSpace<dim,
                                      spacedim,
                                      KernelFunctionType,
                                      KernelNumberType> &f,
  const KernelNumberType                                 factor,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
    &bem_values,
  const PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
              &scratch_data,
  unsigned int component = 0)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  const CellNeighboringType cell_neighboring_type =
    f.get_cell_neighboring_type();
  unsigned int               n_quad_points;
  const Quadrature<dim * 2> *quad_rule;
  switch (cell_neighboring_type)
    {
      case CellNeighboringType::SamePanel:
        n_quad_points = bem_values.quad_rule_for_same_panel.size();
        quad_rule     = &bem_values.quad_rule_for_same_panel;
        break;
      case CellNeighboringType::CommonEdge:
        n_quad_points = bem_values.quad_rule_for_common_edge.size();
        quad_rule     = &bem_values.quad_rule_for_common_edge;
        break;
      case CellNeighboringType::CommonVertex:
        n_quad_points = bem_values.quad_rule_for_common_vertex.size();
        quad_rule     = &bem_values.quad_rule_for_common_vertex;
        break;
      default:
        AssertThrow(false, ExcInternalError());
    }

  KernelNumberType           result       = KernelNumberType();
  const std::vector<double> &quad_weights = quad_rule->get_weights();
  // Iterate over each quadrature point in the 4D quadrature object for the
  // Sauter parametric space.
  for (unsigned int quad_no = 0; quad_no < n_quad_points; quad_no++)
    {
      // The quadrature weight must be multiplied with the value, because when
      // the kernel function is evaluated at the quadrature point, there is no
      // quadrature weight multiplied in the function call @p f.value().
      result += f.value(quad_no, bem_values, scratch_data, component) *
                static_cast<real_type>(quad_weights[quad_no]);
    }

  return result * factor;
}


/**
 * Apply the Sauter's quadrature rule to the kernel function pulled back to
 * the Sauter's parametric space. The result will also be multiplied by a
 * factor.
 *
 * This version is only used for the regular cell neighboring type.
 *
 * @param f
 * @param factor
 * @param kx_cell_index_local
 * @param ky_cell_index_local
 * @param bem_values
 * @param component
 * @return
 */
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType>
KernelNumberType
ApplyQuadratureUsingBEMValues(
  const KernelPulledbackToSauterSpace<dim,
                                      spacedim,
                                      KernelFunctionType,
                                      KernelNumberType> &f,
  const KernelNumberType                                 factor,
  const types::global_cell_index                         kx_cell_index_local,
  const types::global_cell_index                         ky_cell_index_local,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
              &bem_values,
  unsigned int component = 0)
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  Assert(f.get_cell_neighboring_type() == CellNeighboringType::Regular,
         ExcInternalError());
  // This quadrature object is 2D, therefore, double loops are needed to iterate
  // over 4D quadrature points over a pair of cells as below.
  const unsigned int n_quad_points = bem_values.quad_rule_for_regular.size();
  KernelNumberType   result        = KernelNumberType();

  // When the cell neighboring type is regular, we use precomputed cell values
  // (saved as data tables in @p bem_values) to evaluate the quadrature directly.
  // The iteration is performed in double loops.
  for (unsigned int kx_quad_no = 0; kx_quad_no < n_quad_points; kx_quad_no++)
    for (unsigned int ky_quad_no = 0; ky_quad_no < n_quad_points; ky_quad_no++)
      {
        // Quadrature weights have already been multiplied into Jacobian
        // determinants for Kx and Ky during the call of @p f.value(),
        // therefore, they are not needed here.
        result += f.value(kx_cell_index_local,
                          ky_cell_index_local,
                          kx_quad_no,
                          ky_quad_no,
                          bem_values,
                          component);
      }

  return result * factor;
}


/**
 * Perform the Galerkin-BEM double integral with respect to a boundary
 * integral operator (represented as the input kernel function) using
 * Sauter's quadrature for the DoFs in a pair of cells \f$K_x\f$ and
 * \f$K_y\f$.
 *
 * \mynote{When the boundary integral operator is the hyper singular
 * operator, the regularized bilinear form in \f$\mathbb{R}^3\f$ is \f[
 * \left\langle Du,v \right\rangle_{\Gamma} =
 * \frac{1}{4\pi}\int_{\Gamma}\int_{\Gamma}
 * \frac{\underline{\curl}_{\Gamma}u(y)\cdot\underline{\curl}_{\Gamma}v(x)}{\abs{x-y}}
 * ds_x ds_y.
 * \f]
 * It needs special treatment, i.e. calculation of the surface curl of the
 * basis functions for ansatz and test functions.
 *
 * This function is only used in the case when a full matrix for
 * a boundary integral operator is to be constructed. Therefore, this function
 * is only meaningful for algorithm verification. In real application, an
 * \hmatrix should be built. Also note that even for the near field matrix
 * node in an \hmatrix, which must be a full matrix, the Sauter's quadrature
 * is built in the paradigm of "on a pair of DoFs" instead of "on a pair of
 * cells". This is because the two cluster trees associated with an \hmatrix
 * use partition by DoF support points in stead of partition by cells.
 *
 * Usually, this function is called in a double loop: the outer loop iterates
 * over each cell, i.e. \f$K_x\f$, in the test space, while the inner loop
 * iterates over each cell, i.e. \f$K_y\f$, in the ansatz space. Therefore,
 * before calling this function, the mapping support points and DoF indices in
 * \f$K_x\f$ have been calculated in the outer loop and there is no need to
 * compute them again.
 *
 * When this function is called independently for demonstration or
 * verification purpose, the mapping support points and DoF indices in
 * \f$K_x\f$ may not have been computed. This condition is indicated in the
 * flag variable @p is_kx_mapping_internaldata_computed.}
 *
 * @param kernel
 * @param factor
 * @param kx_cell_iter
 * @param ky_cell_iter
 * @param kx_mapping_info
 * @param ky_mapping_info
 * @param bem_values
 * @param normal_detector
 * @param scratch_data
 * @param copy_data
 * @param is_kx_mapping_internaldata_computed
 * @param is_symmetric
 */
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType,
          typename SurfaceNormalDetector>
void
sauter_quadrature_on_one_pair_of_cells(
  const KernelFunctionType<spacedim, KernelNumberType>           &kernel,
  const KernelNumberType                                          kernel_factor,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &kx_cell_iter,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &ky_cell_iter,
  const MappingInfo<dim, spacedim> &kx_mapping_info,
  const MappingInfo<dim, spacedim> &ky_mapping_info,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
                              &bem_values,
  const SurfaceNormalDetector &normal_detector,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                                                           &scratch_data,
  PairCellWisePerTaskData<dim, spacedim, KernelNumberType> &copy_data,
  const bool is_kx_mapping_internaldata_computed = true,
  const bool is_symmetric                        = false)
{
  /**
   * Detect the cell neighboring type based on cell vertex indices.
   */
  CellNeighboringType cell_neighboring_type =
    BEMTools::detect_cell_neighboring_type_for_same_triangulations<dim,
                                                                   spacedim>(
      kx_cell_iter,
      ky_cell_iter,
      scratch_data.common_vertex_pair_local_indices);

  const types::global_cell_index kx_cell_index_global =
    kx_cell_iter->active_cell_index();
  const types::global_cell_index ky_cell_index_global =
    ky_cell_iter->active_cell_index();
  const unsigned int kx_n_dofs = kx_cell_iter->get_fe().dofs_per_cell;
  const unsigned int ky_n_dofs = ky_cell_iter->get_fe().dofs_per_cell;

  if (!is_kx_mapping_internaldata_computed)
    kx_cell_iter->get_dof_indices(
      scratch_data.kx_local_dof_indices_in_default_dof_order);

  ky_cell_iter->get_dof_indices(
    scratch_data.ky_local_dof_indices_in_default_dof_order);

  permute_dofs_and_mapping_support_points_for_sauter_quad(
    scratch_data,
    copy_data,
    cell_neighboring_type,
    kx_cell_iter,
    ky_cell_iter,
    kx_mapping_info,
    ky_mapping_info,
    bem_values.mapping_support_point_table);

  // Get the indices for getting the mapping objects for kx and ky.
  const unsigned int kx_mapping_index =
    kx_mapping_info.get_mapping().get_degree() - 1;
  const unsigned int ky_mapping_index =
    ky_mapping_info.get_mapping().get_degree() - 1;
  const unsigned int kx_mapping_n_shape_functions =
    kx_mapping_info.get_data()->n_shape_functions;
  const unsigned int ky_mapping_n_shape_functions =
    ky_mapping_info.get_data()->n_shape_functions;

  // Check the direction of surface normal vectors for kx and ky. Because kx
  // and ky may belong to different surface entities, their inward/outward
  // properties may be different.
  const bool is_kx_normal_inward =
    normal_detector.is_normal_vector_inward(kx_cell_iter->material_id());
  const bool is_ky_normal_inward =
    normal_detector.is_normal_vector_inward(ky_cell_iter->material_id());

  switch (cell_neighboring_type)
    {
      case CellNeighboringType::SamePanel:
        compute_cell_values_for_all_shape_functions_same_panel(
          kx_cell_index_global,
          ky_cell_index_global,
          kx_mapping_index,
          ky_mapping_index,
          kx_mapping_n_shape_functions,
          ky_mapping_n_shape_functions,
          is_kx_normal_inward,
          is_ky_normal_inward,
          bem_values,
          scratch_data);

        break;
      case CellNeighboringType::CommonEdge:
        compute_cell_values_for_all_shape_functions_common_edge(
          kx_mapping_index,
          ky_mapping_index,
          kx_mapping_n_shape_functions,
          ky_mapping_n_shape_functions,
          is_kx_normal_inward,
          is_ky_normal_inward,
          bem_values,
          scratch_data);

        break;
      case CellNeighboringType::CommonVertex:
        compute_cell_values_for_all_shape_functions_common_vertex(
          kx_mapping_index,
          ky_mapping_index,
          kx_mapping_n_shape_functions,
          ky_mapping_n_shape_functions,
          is_kx_normal_inward,
          is_ky_normal_inward,
          bem_values,
          scratch_data);

        break;
      default:
        break;
    }

  //  Clear the local matrix in case that it is reused from another
  //  finished task. N.B. Its memory has already been allocated in the
  //  constructor of @p CellPairWisePerTaskData.
  copy_data.local_pair_cell_matrix.reinit(
    copy_data.kx_local_dof_indices_permuted.size(),
    copy_data.ky_local_dof_indices_permuted.size());

  // Iterate over DoFs for the test function space in \f$K_x\f$.
  for (unsigned int i = 0; i < kx_n_dofs; i++)
    {
      // Iterate over DoFs for the trial function space in \f$K_y\f$.
      for (unsigned int j = 0; j < ky_n_dofs; j++)
        {
          // When the bilinear form (of course the BEM kernel function) is
          // symmetric, we only compute the matrix entry whose global row DoF
          // index >= global column DoF index, i.e. the diagonal and lower
          // triangular parts of the global full matrix are computed.
          //
          // @p i and @p j are row and column indices for the local pair
          // cellwise matrix, while @p copy_data.kx_local_dof_indices_permuted[i]
          // and @p copy_data.ky_local_dof_indices_permuted[j] are their global
          // DoF indices.
          if (!is_symmetric ||
              (is_symmetric && copy_data.kx_local_dof_indices_permuted[i] >=
                                 copy_data.ky_local_dof_indices_permuted[j]))
            {
              // Pullback the kernel function to unit cell.
              KernelPulledbackToUnitCell<dim,
                                         spacedim,
                                         KernelFunctionType,
                                         KernelNumberType>
                kernel_pullback_on_unit(kernel, cell_neighboring_type, i, j);

              // Pullback the kernel function to Sauter parameter space.
              KernelPulledbackToSauterSpace<dim,
                                            spacedim,
                                            KernelFunctionType,
                                            KernelNumberType>
                kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                          cell_neighboring_type);

              // Apply the Sauter numerical quadrature.
              if (cell_neighboring_type != CellNeighboringType::Regular)
                {
                  copy_data.local_pair_cell_matrix(i, j) =
                    ApplyQuadratureUsingBEMValues(kernel_pullback_on_sauter,
                                                  kernel_factor,
                                                  bem_values,
                                                  scratch_data);
                }
              else
                {
                  // At the moment, the full matrix solver only supports
                  // building matrices on the whole triangulation, not on a
                  // subdomain. Therefore, local cell indices related to a
                  // bilinear form are just the same as global cell indices.
                  copy_data.local_pair_cell_matrix(i, j) =
                    ApplyQuadratureUsingBEMValues(kernel_pullback_on_sauter,
                                                  kernel_factor,
                                                  kx_cell_index_global,
                                                  ky_cell_index_global,
                                                  bem_values);
                }
            }
        }
    }
}


/**
 * This overloaded version is used for WorkStream parallelization over
 * \f$K_y\f$ cells.
 */
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType,
          typename SurfaceNormalDetector>
void
sauter_quadrature_on_one_pair_of_cells_parallel_over_ky(
  const KernelFunctionType<spacedim, KernelNumberType>           &kernel,
  const KernelNumberType                                          kernel_factor,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &kx_cell_iter,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &ky_cell_iter,
  const std::vector<MappingInfo<dim, spacedim> *>                &mappings,
  const std::map<types::material_id, unsigned int>
                                   &material_id_to_mapping_index,
  const MappingInfo<dim, spacedim> &kx_mapping_info,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
                              &bem_values,
  const SurfaceNormalDetector &normal_detector,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                                                           &scratch_data,
  PairCellWisePerTaskData<dim, spacedim, KernelNumberType> &copy_data,
  const bool is_kx_mapping_internaldata_computed = true,
  const bool is_symmetric                        = false)
{
  const unsigned int ky_mapping_index =
    material_id_to_mapping_index.at(ky_cell_iter->material_id());
  MappingInfo<dim, spacedim> &ky_mapping_info = *mappings[ky_mapping_index];

  sauter_quadrature_on_one_pair_of_cells(kernel,
                                         kernel_factor,
                                         kx_cell_iter,
                                         ky_cell_iter,
                                         kx_mapping_info,
                                         ky_mapping_info,
                                         bem_values,
                                         normal_detector,
                                         scratch_data,
                                         copy_data,
                                         is_kx_mapping_internaldata_computed,
                                         is_symmetric);
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename KernelNumberType,
          typename SurfaceNormalDetector>
KernelNumberType
sauter_quadrature_on_one_pair_of_shape_functions(
  const KernelFunctionType<spacedim, KernelNumberType>           &kernel,
  const KernelNumberType                                          kernel_factor,
  const unsigned int                                              i,
  const unsigned int                                              j,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &kx_cell_iter,
  const typename DoFHandler<dim, spacedim>::active_cell_iterator &ky_cell_iter,
  const MappingInfo<dim, spacedim> &kx_mapping_info,
  const MappingInfo<dim, spacedim> &ky_mapping_info,
  const BEMValues<dim,
                  spacedim,
                  typename numbers::NumberTraits<KernelNumberType>::real_type>
                              &bem_values,
  const SurfaceNormalDetector &normal_detector,
  PairCellWiseScratchDataForFullMatrix<dim, spacedim, KernelNumberType>
                                                           &scratch_data,
  PairCellWisePerTaskData<dim, spacedim, KernelNumberType> &copy_data)
{
  AssertIndexRange(i, kx_cell_iter->get_fe().dofs_per_cell);
  AssertIndexRange(j, ky_cell_iter->get_fe().dofs_per_cell);

  /**
   * Detect the cell neighboring type based on cell vertex indices.
   */
  CellNeighboringType cell_neighboring_type =
    BEMTools::detect_cell_neighboring_type_for_same_triangulations<dim,
                                                                   spacedim>(
      kx_cell_iter,
      ky_cell_iter,
      scratch_data.common_vertex_pair_local_indices);

  const types::global_cell_index kx_cell_index_global =
    kx_cell_iter->active_cell_index();
  const types::global_cell_index ky_cell_index_global =
    ky_cell_iter->active_cell_index();

  kx_cell_iter->get_dof_indices(
    scratch_data.kx_local_dof_indices_in_default_dof_order);
  ky_cell_iter->get_dof_indices(
    scratch_data.ky_local_dof_indices_in_default_dof_order);

  permute_dofs_and_mapping_support_points_for_sauter_quad(
    scratch_data,
    copy_data,
    cell_neighboring_type,
    kx_cell_iter,
    ky_cell_iter,
    kx_mapping_info,
    ky_mapping_info,
    bem_values.mapping_support_point_table);

  // Get the indices for getting the mapping objects for kx and ky.
  const unsigned int kx_mapping_index =
    kx_mapping_info.get_mapping().get_degree() - 1;
  const unsigned int ky_mapping_index =
    ky_mapping_info.get_mapping().get_degree() - 1;
  const unsigned int kx_mapping_n_shape_functions =
    kx_mapping_info.get_data()->n_shape_functions;
  const unsigned int ky_mapping_n_shape_functions =
    ky_mapping_info.get_data()->n_shape_functions;

  // Check the direction of surface normal vectors for kx and ky. Because kx
  // and ky may belong to different surface entities, their inward/outward
  // properties may be different.
  const bool is_kx_normal_inward =
    normal_detector.is_normal_vector_inward(kx_cell_iter->material_id());
  const bool is_ky_normal_inward =
    normal_detector.is_normal_vector_inward(ky_cell_iter->material_id());

  switch (cell_neighboring_type)
    {
      case CellNeighboringType::SamePanel:
        compute_cell_values_for_one_pair_of_shape_functions_same_panel(
          i,
          j,
          kx_cell_index_global,
          ky_cell_index_global,
          kx_mapping_index,
          ky_mapping_index,
          kx_mapping_n_shape_functions,
          ky_mapping_n_shape_functions,
          is_kx_normal_inward,
          is_ky_normal_inward,
          bem_values,
          scratch_data);

        break;
      case CellNeighboringType::CommonEdge:
        compute_cell_values_for_one_pair_of_shape_functions_common_edge(
          i,
          j,
          kx_mapping_index,
          ky_mapping_index,
          kx_mapping_n_shape_functions,
          ky_mapping_n_shape_functions,
          is_kx_normal_inward,
          is_ky_normal_inward,
          bem_values,
          scratch_data);

        break;
      case CellNeighboringType::CommonVertex:
        compute_cell_values_for_one_pair_of_shape_functions_common_vertex(
          i,
          j,
          kx_mapping_index,
          ky_mapping_index,
          kx_mapping_n_shape_functions,
          ky_mapping_n_shape_functions,
          is_kx_normal_inward,
          is_ky_normal_inward,
          bem_values,
          scratch_data);

        break;
      default:
        break;
    }

  // Pullback the kernel function to unit cell.
  KernelPulledbackToUnitCell<dim,
                             spacedim,
                             KernelFunctionType,
                             KernelNumberType>
    kernel_pullback_on_unit(kernel, cell_neighboring_type, i, j);

  // Pullback the kernel function to Sauter parameter space.
  KernelPulledbackToSauterSpace<dim,
                                spacedim,
                                KernelFunctionType,
                                KernelNumberType>
    kernel_pullback_on_sauter(kernel_pullback_on_unit, cell_neighboring_type);

  // Apply the Sauter numerical quadrature.
  if (cell_neighboring_type != CellNeighboringType::Regular)
    return ApplyQuadratureUsingBEMValues(kernel_pullback_on_sauter,
                                         kernel_factor,
                                         bem_values,
                                         scratch_data);
  else
    // At the moment, the full matrix solver only supports building matrices on
    // the whole triangulation, not on a subdomain. Therefore, local cell
    // indices related to a bilinear form are just the same as global cell
    // indices.
    return ApplyQuadratureUsingBEMValues(kernel_pullback_on_sauter,
                                         kernel_factor,
                                         kx_cell_index_global,
                                         ky_cell_index_global,
                                         bem_values);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_H_
