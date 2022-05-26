/**
 * @file sauter_quadrature.h
 * @brief Introduction of sauter_quadrature.h
 *
 * @date 2022-03-02
 * @author Jihuan Tian
 */
#ifndef INCLUDE_SAUTER_QUADRATURE_H_
#define INCLUDE_SAUTER_QUADRATURE_H_


#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/full_matrix.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include "bem_kernels.h"
#include "bem_values.h"
#include "sauter_quadrature_tools.h"

namespace IdeoBEM
{
  using namespace dealii;

  /**
   * Build the topology for "DoF support point to cell" relation.
   *
   * @param dof_to_cell_topo
   * @param dof_handler
   * @param fe_index
   */
  template <int dim, int spacedim>
  void
  build_dof_to_cell_topology(
    std::vector<std::vector<unsigned int>> &dof_to_cell_topo,
    const DoFHandler<dim, spacedim> &       dof_handler,
    const unsigned int                      fe_index = 0)
  {
    const types::global_dof_index        n_dofs = dof_handler.n_dofs();
    const FiniteElement<dim, spacedim> & fe     = dof_handler.get_fe(fe_index);
    const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    dof_to_cell_topo.resize(n_dofs);

    /**
     * Iterate over each active cell in the triangulation and extract the DoF
     * indices.
     */
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &cell :
         dof_handler.active_cell_iterators())
      {
        cell->get_dof_indices(local_dof_indices);
        for (auto dof_index : local_dof_indices)
          {
            dof_to_cell_topo[dof_index].push_back(cell->active_cell_index());
          }
      }
  }


  /**
   * Print out the topological information about DoF support point to cell
   * relation.
   *
   * @param dof_to_cell_topo
   */
  void
  print_dof_to_cell_topology(
    const std::vector<std::vector<unsigned int>> &dof_to_cell_topo);


  /**
   * Get the DoF indices associated with the cell vertices or corners from a
   * list of DoF indices which have been arranged in either the forward or
   * backward lexicographic order. The results are returned in an array as the
   * function's return value.
   *
   * \mynote{There are <code>GeometryInfo<dim>::vertices_per_cell</code>
   * vertices in the returned array, among which the last two vertex DoF indices
   * have been swapped in this function so that the whole list of vertex DoF
   * indices in the returned array are arranged in either the clockwise or
   * counter clockwise order instead of the original zigzag order.}
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
    const FiniteElement<dim, spacedim> &        fe,
    const std::vector<types::global_dof_index> &dof_indices)
  {
    Assert(dim == 2, ExcNotImplemented());

    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      vertex_dof_indices;

    vertex_dof_indices[0] = dof_indices[0];
    vertex_dof_indices[1] = dof_indices[fe.dofs_per_face - 1];
    vertex_dof_indices[2] = dof_indices[dof_indices.size() - 1];
    vertex_dof_indices[3] =
      dof_indices[dof_indices.size() - 1 - (fe.dofs_per_face - 1)];

    return vertex_dof_indices;
  }


  /**
   * Get the DoF indices associated with the cell vertices or corners from a
   * list of DoF indices which have been arranged in either the forward or
   * backward lexicographic order. The results are returned in an array as the
   * last argument of this function.
   *
   * \mynote{There are <code>GeometryInfo<dim>::vertices_per_cell</code>
   * vertices in the returned array, among which the last two vertex DoF indices
   * have been swapped in this function so that the whole list of vertex DoF
   * indices in the returned array are arranged in either the clockwise or
   * counter clockwise order instead of the original zigzag order.}
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
    const FiniteElement<dim, spacedim> &        fe,
    const std::vector<types::global_dof_index> &dof_indices,
    std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
      &vertex_dof_indices)
  {
    Assert(dim == 2, ExcNotImplemented());

    vertex_dof_indices[0] = dof_indices[0];
    vertex_dof_indices[1] = dof_indices[fe.dofs_per_face - 1];
    vertex_dof_indices[2] = dof_indices[dof_indices.size() - 1];
    vertex_dof_indices[3] =
      dof_indices[dof_indices.size() - 1 - (fe.dofs_per_face - 1)];
  }


  /**
   * Get the DoF index for the starting vertex in the standard configuration of
   * the cell pair aimed for the Galerkin BEM double integration using the
   * Sauter's method.
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
   * @param vertex_dof_index_intersection The vector storing the intersection
   * of vertex DoF indices for \f$K_x\f$ and \f$K_y\f$.
   * @param local_vertex_dof_indices_swapped Vertex DoF indices with the last
   * two swapped, which have been obtained from the function
   * @p get_vertex_dof_indices_swapped.
   * @return The array index for the starting vertex, wrt. the original list of
   * vertex DoF indices, i.e. the last two elements of which are not swapped.
   */
  template <int vertices_per_cell>
  unsigned int
  get_start_vertex_dof_index(
    const std::vector<types::global_dof_index> &vertex_dof_index_intersection,
    const std::array<types::global_dof_index, vertices_per_cell>
      &local_vertex_dof_indices_swapped)
  {
    unsigned int starting_vertex_index = 9999;

    switch (vertex_dof_index_intersection.size())
      {
        case 2: // Common edge case
          {
            typename std::array<types::global_dof_index,
                                vertices_per_cell>::const_iterator
              first_common_vertex_iterator =
                std::find(local_vertex_dof_indices_swapped.cbegin(),
                          local_vertex_dof_indices_swapped.cend(),
                          vertex_dof_index_intersection[0]);
            typename std::array<types::global_dof_index,
                                vertices_per_cell>::const_iterator
              second_common_vertex_iterator =
                std::find(local_vertex_dof_indices_swapped.cbegin(),
                          local_vertex_dof_indices_swapped.cend(),
                          vertex_dof_index_intersection[1]);

            if ((first_common_vertex_iterator + 1) !=
                local_vertex_dof_indices_swapped.cend())
              {
                if (*(first_common_vertex_iterator + 1) ==
                    vertex_dof_index_intersection[1])
                  {
                    starting_vertex_index =
                      first_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
                else
                  {
                    starting_vertex_index =
                      second_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
              }
            else
              {
                if ((*local_vertex_dof_indices_swapped.cbegin()) ==
                    vertex_dof_index_intersection[1])
                  {
                    starting_vertex_index =
                      first_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
                else
                  {
                    starting_vertex_index =
                      second_common_vertex_iterator -
                      local_vertex_dof_indices_swapped.cbegin();
                  }
              }

            break;
          }
        case 1: // Common vertex case
          {
            typename std::array<types::global_dof_index,
                                vertices_per_cell>::const_iterator
              first_common_vertex_iterator =
                std::find(local_vertex_dof_indices_swapped.cbegin(),
                          local_vertex_dof_indices_swapped.cend(),
                          vertex_dof_index_intersection[0]);
            Assert(first_common_vertex_iterator !=
                     local_vertex_dof_indices_swapped.cend(),
                   ExcInternalError());

            starting_vertex_index = first_common_vertex_iterator -
                                    local_vertex_dof_indices_swapped.cbegin();

            break;
          }
        default:
          Assert(false, ExcInternalError());
          break;
      }

    /**
     * Because the last two elements in the original list of vertex DoF indices
     * have been swapped, we need to correct the starting vertex index when it
     * is one of the last two elements.
     */
    if (starting_vertex_index == 2)
      {
        starting_vertex_index = 3;
      }
    else if (starting_vertex_index == 3)
      {
        starting_vertex_index = 2;
      }

    return starting_vertex_index;
  }


  /**
   * TODO: Permute DoFs support points in real cells and their associated global
   * DoF indices for Sauter quadrature, the behavior of which depends on the
   * detected cell neighboring types.
   *
   * \mynote{This version does not involve @p PairCellWiseScratchData and
   * @p PairCellWisePerTaskData.}
   *
   * @param kx_support_points_permuted
   * @param ky_support_points_permuted
   * @param kx_local_dof_indices_permuted
   * @param ky_local_dof_indices_permuted
   * @param kx_cell_iter
   * @param ky_cell_iter
   * @param kx_mapping
   * @param ky_mapping
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  permute_dofs_for_sauter_quad(
    std::vector<Point<spacedim>> &        kx_support_points_permuted,
    std::vector<Point<spacedim>> &        ky_support_points_permuted,
    std::vector<types::global_dof_index> &kx_local_dof_indices_permuted,
    std::vector<types::global_dof_index> &ky_local_dof_indices_permuted,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {}


  /**
   * Permute DoFs support points in real cells and their associated global
   * DoF indices for Sauter quadrature, the behavior of which depends on the
   * detected cell neighboring types.
   *
   * \mynote{This version involves @p PairCellWiseScratchData and
   * @p PairCellWisePerTaskData.}
   *
   * @param scratch
   * @param data
   * @param kx_cell_iter
   * @param ky_cell_iter
   * @param kx_mapping
   * @param ky_mapping
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  permute_dofs_for_sauter_quad(
    PairCellWiseScratchData & scratch,
    PairCellWisePerTaskData & data,
    const CellNeighboringType cell_neighboring_type,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
    // hierarchical order.
    get_hierarchic_support_points_in_real_cell(
      kx_cell_iter, kx_fe, kx_mapping, scratch.kx_support_points_hierarchical);
    get_hierarchic_support_points_in_real_cell(
      ky_cell_iter, ky_fe, ky_mapping, scratch.ky_support_points_hierarchical);

    // N.B. The vector holding local DoF indices has to have the right size
    // before being passed to the function <code>get_dof_indices</code>.
    kx_cell_iter->get_dof_indices(scratch.kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(scratch.ky_local_dof_indices_hierarchical);

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(scratch.vertex_dof_index_intersection.size() ==
                     vertices_per_cell,
                   ExcInternalError());

            /**
             * Get support points in the lexicographic order.
             */
            permute_vector(scratch.kx_support_points_hierarchical,
                           scratch.kx_fe_poly_space_numbering_inverse,
                           scratch.kx_support_points_permuted);
            permute_vector(scratch.ky_support_points_hierarchical,
                           scratch.ky_fe_poly_space_numbering_inverse,
                           scratch.ky_support_points_permuted);

            /**
             * Get permuted local DoF indices in the lexicographic order.
             */
            permute_vector(scratch.kx_local_dof_indices_hierarchical,
                           scratch.kx_fe_poly_space_numbering_inverse,
                           data.kx_local_dof_indices_permuted);
            permute_vector(scratch.ky_local_dof_indices_hierarchical,
                           scratch.ky_fe_poly_space_numbering_inverse,
                           data.ky_local_dof_indices_permuted);

            break;
          }
        case CommonEdge:
          {
            // This part handles the common edge case of Sauter's
            // quadrature rule.
            // 1. Get the DoF indices in the lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in the reversed lexicographic order
            // for \f$K_y\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
            // swapped, such that the four vertices are in clockwise or
            // counter clockwise order.
            // 4. Determine the starting vertex.

            Assert(scratch.vertex_dof_index_intersection.size() ==
                     GeometryInfo<2>::vertices_per_face,
                   ExcInternalError());

            permute_vector(scratch.kx_local_dof_indices_hierarchical,
                           scratch.kx_fe_poly_space_numbering_inverse,
                           data.kx_local_dof_indices_permuted);

            permute_vector(scratch.ky_local_dof_indices_hierarchical,
                           scratch.ky_fe_reversed_poly_space_numbering_inverse,
                           data.ky_local_dof_indices_permuted);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped;
            get_vertex_dof_indices_swapped<2, 3>(
              kx_fe,
              data.kx_local_dof_indices_permuted,
              kx_local_vertex_dof_indices_swapped);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped;
            get_vertex_dof_indices_swapped<2, 3>(
              ky_fe,
              data.ky_local_dof_indices_permuted,
              ky_local_vertex_dof_indices_swapped);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                scratch.vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                scratch.vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            generate_forward_dof_permutation(kx_fe,
                                             kx_starting_vertex_index,
                                             scratch.kx_local_dof_permutation);
            generate_backward_dof_permutation(ky_fe,
                                              ky_starting_vertex_index,
                                              scratch.ky_local_dof_permutation);

            permute_vector(scratch.kx_support_points_hierarchical,
                           scratch.kx_local_dof_permutation,
                           scratch.kx_support_points_permuted);
            permute_vector(scratch.ky_support_points_hierarchical,
                           scratch.ky_local_dof_permutation,
                           scratch.ky_support_points_permuted);

            permute_vector(scratch.kx_local_dof_indices_hierarchical,
                           scratch.kx_local_dof_permutation,
                           data.kx_local_dof_indices_permuted);
            permute_vector(scratch.ky_local_dof_indices_hierarchical,
                           scratch.ky_local_dof_permutation,
                           data.ky_local_dof_indices_permuted);

            break;
          }
        case CommonVertex:
          {
            Assert(scratch.vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            permute_vector(scratch.kx_local_dof_indices_hierarchical,
                           scratch.kx_fe_poly_space_numbering_inverse,
                           data.kx_local_dof_indices_permuted);

            permute_vector(scratch.ky_local_dof_indices_hierarchical,
                           scratch.ky_fe_poly_space_numbering_inverse,
                           data.ky_local_dof_indices_permuted);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped;
            get_vertex_dof_indices_swapped<2, 3>(
              kx_fe,
              data.kx_local_dof_indices_permuted,
              kx_local_vertex_dof_indices_swapped);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped;
            get_vertex_dof_indices_swapped<2, 3>(
              ky_fe,
              data.ky_local_dof_indices_permuted,
              ky_local_vertex_dof_indices_swapped);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                scratch.vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                scratch.vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            generate_forward_dof_permutation(kx_fe,
                                             kx_starting_vertex_index,
                                             scratch.kx_local_dof_permutation);
            generate_forward_dof_permutation(ky_fe,
                                             ky_starting_vertex_index,
                                             scratch.ky_local_dof_permutation);

            permute_vector(scratch.kx_support_points_hierarchical,
                           scratch.kx_local_dof_permutation,
                           scratch.kx_support_points_permuted);
            permute_vector(scratch.ky_support_points_hierarchical,
                           scratch.ky_local_dof_permutation,
                           scratch.ky_support_points_permuted);

            permute_vector(scratch.kx_local_dof_indices_hierarchical,
                           scratch.kx_local_dof_permutation,
                           data.kx_local_dof_indices_permuted);
            permute_vector(scratch.ky_local_dof_indices_hierarchical,
                           scratch.ky_local_dof_permutation,
                           data.ky_local_dof_indices_permuted);

            break;
          }
        case Regular:
          {
            Assert(scratch.vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            permute_vector(scratch.kx_local_dof_indices_hierarchical,
                           scratch.kx_fe_poly_space_numbering_inverse,
                           data.kx_local_dof_indices_permuted);

            permute_vector(scratch.ky_local_dof_indices_hierarchical,
                           scratch.ky_fe_poly_space_numbering_inverse,
                           data.ky_local_dof_indices_permuted);

            permute_vector(scratch.kx_support_points_hierarchical,
                           scratch.kx_fe_poly_space_numbering_inverse,
                           scratch.kx_support_points_permuted);
            permute_vector(scratch.ky_support_points_hierarchical,
                           scratch.ky_fe_poly_space_numbering_inverse,
                           scratch.ky_support_points_permuted);

            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }
  }


  /**
   * Precalculate surface Jacobians and normal vectors for Sauter quadrature.
   *
   * \mynote{This version involves @p PairCellWiseScratchData.}
   *
   * @param scratch
   * @param data
   * @param cell_neighboring_type
   * @param active_quad_rule
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  calc_jacobian_normals_for_sauter_quad(
    PairCellWiseScratchData &                        scratch,
    const CellNeighboringType                        cell_neighboring_type,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const QGauss<dim * 2> &                          active_quad_rule)
  {
    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(scratch.vertex_dof_index_intersection.size() ==
                     GeometryInfo<dim>::vertices_per_cell,
                   ExcInternalError());

            /**
             * Precalculate surface Jacobians and normal vectors at each
             * quadrature point in the current pair of cells.
             * \mynote{They are stored in the @p scratch data.}
             */
            for (unsigned int k3_index = 0; k3_index < 8; k3_index++)
              {
                for (unsigned int q = 0; q < active_quad_rule.size(); q++)
                  {
                    scratch.kx_jacobians_same_panel(k3_index, q) =
                      surface_jacobian_det_and_normal_vector(
                        k3_index,
                        q,
                        bem_values.kx_shape_grad_matrix_table_for_same_panel,
                        scratch.kx_support_points_permuted,
                        scratch.kx_normals_same_panel(k3_index, q));

                    scratch.ky_jacobians_same_panel(k3_index, q) =
                      surface_jacobian_det_and_normal_vector(
                        k3_index,
                        q,
                        bem_values.ky_shape_grad_matrix_table_for_same_panel,
                        scratch.ky_support_points_permuted,
                        scratch.ky_normals_same_panel(k3_index, q));

                    scratch.kx_quad_points_same_panel(k3_index, q) =
                      transform_unit_to_permuted_real_cell(
                        k3_index,
                        q,
                        bem_values.kx_shape_value_table_for_same_panel,
                        scratch.kx_support_points_permuted);

                    scratch.ky_quad_points_same_panel(k3_index, q) =
                      transform_unit_to_permuted_real_cell(
                        k3_index,
                        q,
                        bem_values.ky_shape_value_table_for_same_panel,
                        scratch.ky_support_points_permuted);
                  }
              }

            break;
          }
        case CommonEdge:
          {
            Assert(scratch.vertex_dof_index_intersection.size() ==
                     GeometryInfo<2>::vertices_per_face,
                   ExcInternalError());

            // Precalculate surface Jacobians and normal vectors.
            for (unsigned int k3_index = 0; k3_index < 6; k3_index++)
              {
                for (unsigned int q = 0; q < active_quad_rule.size(); q++)
                  {
                    scratch.kx_jacobians_common_edge(k3_index, q) =
                      surface_jacobian_det_and_normal_vector(
                        k3_index,
                        q,
                        bem_values.kx_shape_grad_matrix_table_for_common_edge,
                        scratch.kx_support_points_permuted,
                        scratch.kx_normals_common_edge(k3_index, q));

                    scratch.ky_jacobians_common_edge(k3_index, q) =
                      surface_jacobian_det_and_normal_vector(
                        k3_index,
                        q,
                        bem_values.ky_shape_grad_matrix_table_for_common_edge,
                        scratch.ky_support_points_permuted,
                        scratch.ky_normals_common_edge(k3_index, q));

                    scratch.kx_quad_points_common_edge(k3_index, q) =
                      transform_unit_to_permuted_real_cell(
                        k3_index,
                        q,
                        bem_values.kx_shape_value_table_for_common_edge,
                        scratch.kx_support_points_permuted);

                    scratch.ky_quad_points_common_edge(k3_index, q) =
                      transform_unit_to_permuted_real_cell(
                        k3_index,
                        q,
                        bem_values.ky_shape_value_table_for_common_edge,
                        scratch.ky_support_points_permuted);
                  }
              }

            break;
          }
        case CommonVertex:
          {
            Assert(scratch.vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            // Precalculate surface Jacobians and normal vectors.
            for (unsigned int k3_index = 0; k3_index < 4; k3_index++)
              {
                for (unsigned int q = 0; q < active_quad_rule.size(); q++)
                  {
                    scratch.kx_jacobians_common_vertex(k3_index, q) =
                      surface_jacobian_det_and_normal_vector(
                        k3_index,
                        q,
                        bem_values.kx_shape_grad_matrix_table_for_common_vertex,
                        scratch.kx_support_points_permuted,
                        scratch.kx_normals_common_vertex(k3_index, q));

                    scratch.ky_jacobians_common_vertex(k3_index, q) =
                      surface_jacobian_det_and_normal_vector(
                        k3_index,
                        q,
                        bem_values.ky_shape_grad_matrix_table_for_common_vertex,
                        scratch.ky_support_points_permuted,
                        scratch.ky_normals_common_vertex(k3_index, q));

                    scratch.kx_quad_points_common_vertex(k3_index, q) =
                      transform_unit_to_permuted_real_cell(
                        k3_index,
                        q,
                        bem_values.kx_shape_value_table_for_common_vertex,
                        scratch.kx_support_points_permuted);

                    scratch.ky_quad_points_common_vertex(k3_index, q) =
                      transform_unit_to_permuted_real_cell(
                        k3_index,
                        q,
                        bem_values.ky_shape_value_table_for_common_vertex,
                        scratch.ky_support_points_permuted);
                  }
              }

            break;
          }
        case Regular:
          {
            Assert(scratch.vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            // Precalculate surface Jacobians and normal vectors.
            for (unsigned int q = 0; q < active_quad_rule.size(); q++)
              {
                scratch.kx_jacobians_regular(0, q) =
                  surface_jacobian_det_and_normal_vector(
                    0,
                    q,
                    bem_values.kx_shape_grad_matrix_table_for_regular,
                    scratch.kx_support_points_permuted,
                    scratch.kx_normals_regular(0, q));

                scratch.ky_jacobians_regular(0, q) =
                  surface_jacobian_det_and_normal_vector(
                    0,
                    q,
                    bem_values.ky_shape_grad_matrix_table_for_regular,
                    scratch.ky_support_points_permuted,
                    scratch.ky_normals_regular(0, q));

                scratch.kx_quad_points_regular(0, q) =
                  transform_unit_to_permuted_real_cell(
                    0,
                    q,
                    bem_values.kx_shape_value_table_for_regular,
                    scratch.kx_support_points_permuted);

                scratch.ky_quad_points_regular(0, q) =
                  transform_unit_to_permuted_real_cell(
                    0,
                    q,
                    bem_values.ky_shape_value_table_for_regular,
                    scratch.ky_support_points_permuted);
              }

            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }
  }


  /**
   * Perform Sauter's quadrature rule on a pair of quadrangular cells, which
   * handles various cases including same panel, common edge, common vertex and
   * regular cell neighboring types. This functions returns the computed local
   * matrix without assembling it to the global matrix.
   *
   * \mynote{In this version, shape function values and their gradient values
   * are not precalculated.}
   *
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to \f$K_x\f$.
   * @param kx_cell_iter Iterator pointing to \f$K_y\f$.
   * @param kx_mapping Mapping used for \f$K_x\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param kx_mapping Mapping used for \f$K_y\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @return Local matrix
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  FullMatrix<RangeNumberType>
  SauterQuadRule(
    const KernelFunction<spacedim, RangeNumberType> &        kernel_function,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    // N.B. The vector holding local DoF indices has to have the right size
    // before being passed to the function <code>get_dof_indices</code>.
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    // Generate 4D Gauss-Legendre quadrature rules for various cell
    // neighboring types.
    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<4> quad_rule_for_regular(quad_order_for_regular);

    // Local matrix
    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    // Polynomial space inverse numbering for recovering the lexicographic
    // order.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in lexicographic order.
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            // Get permuted local DoF indices.
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_same_panel,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        case CommonEdge:
          {
            // This part handles the common edge case of Sauter's
            // quadrature rule.
            // 1. Get the DoF indices in lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in reversed lexicographic order for
            // \f$K_x\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
            // swapped, such that the four vertices are in clockwise or
            // counter clockwise order.
            // 4. Determine the starting vertex.

            Assert(vertex_dof_index_intersection.size() ==
                     GeometryInfo<dim>::vertices_per_face,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            std::vector<unsigned int>
              ky_fe_reversed_poly_space_numbering_inverse =
                generate_backward_dof_permutation(ky_fe, 0);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_reversed_poly_space_numbering_inverse);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_backward_dof_permutation(ky_fe,
                                                ky_starting_vertex_index);

            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_common_edge,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        case CommonVertex:
          {
            Assert(vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_common_vertex,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        case Regular:
          {
            Assert(vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_regular,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }

    return cell_matrix;
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  FullMatrix<RangeNumberType>
  SauterQuadRule(
    const KernelFunction<spacedim, RangeNumberType> &        kernel_function,
    const BEMValues<dim, spacedim> &                         bem_values,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    // N.B. The vector holding local DoF indices has to have the right size
    // before being passed to the function <code>get_dof_indices</code>.
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    // Polynomial space inverse numbering for recovering the tensor
    // product ordering.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    // Quadrature rule to be adopted depending on the cell neighboring
    // type.
    const QGauss<4> *active_quad_rule;

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in lexicographic order.
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            // Get permuted local DoF indices.
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            active_quad_rule = &(bem_values.quad_rule_for_same_panel);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        case CommonEdge:
          {
            // This part handles the common edge case of Sauter's
            // quadrature rule.
            // 1. Get the DoF indices in lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in reversed lexicographic order for
            // \f$K_x\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
            // swapped, such that the four vertices are in clockwise or
            // counter clockwise order.
            // 4. Determine the starting vertex.

            Assert(vertex_dof_index_intersection.size() ==
                     GeometryInfo<dim>::vertices_per_face,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            std::vector<unsigned int>
              ky_fe_reversed_poly_space_numbering_inverse =
                generate_backward_dof_permutation(ky_fe, 0);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_reversed_poly_space_numbering_inverse);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_backward_dof_permutation(ky_fe,
                                                ky_starting_vertex_index);

            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);

            active_quad_rule = &(bem_values.quad_rule_for_common_edge);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        case CommonVertex:
          {
            Assert(vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);

            active_quad_rule = &(bem_values.quad_rule_for_common_vertex);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        case Regular:
          {
            Assert(vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            active_quad_rule = &(bem_values.quad_rule_for_regular);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }

    // Iterate over DoFs for test function space in lexicographic
    // order in \f$K_x\f$.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        // Iterate over DoFs for ansatz function space in tensor
        // product order in \f$K_y\f$.
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            // Pullback the kernel function to unit cell.
            KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
              kernel_pullback_on_unit(kernel_function,
                                      cell_neighboring_type,
                                      kx_support_points_permuted,
                                      ky_support_points_permuted,
                                      kx_fe,
                                      ky_fe,
                                      &bem_values,
                                      i,
                                      j);

            // Pullback the kernel function to Sauter parameter
            // space.
            KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>
              kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                        cell_neighboring_type,
                                        &bem_values);

            // Apply 4d Sauter numerical quadrature.
            cell_matrix(i, j) =
              ApplyQuadratureUsingBEMValues(*active_quad_rule,
                                            kernel_pullback_on_sauter);
          }
      }

    return cell_matrix;
  }


  /**
   * Perform Sauter's quadrature rule on a pair of quadrangular cells, which
   * handles various cases including same panel, common edge, common vertex and
   * regular cell neighboring types. The computed local matrix values will be
   * assembled into the system matrix, which is passed as the first argument.
   *
   * \mynote{In this version, shape function values and their gradient values
   * are not precalculated.}
   *
   * @param system_matrix The global full matrix to which the local matrix will
   * be assembled.
   * @param kernel_function Laplace kernel function.
   * @param kx_cell_iter Iterator pointing to \f$K_x\f$.
   * @param kx_cell_iter Iterator pointing to \f$K_y\f$.
   * @param kx_mapping Mapping used for \f$K_x\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param ky_mapping Mapping used for \f$K_y\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  SauterQuadRule(
    FullMatrix<RangeNumberType> &                            system_matrix,
    const KernelFunction<spacedim, RangeNumberType> &        kernel_function,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);


    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(kx_cell_iter, kx_fe, kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      hierarchical_support_points_in_real_cell(ky_cell_iter, ky_fe, ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    // N.B. These vectors should have the right size before being passed to
    // the function <code>get_dof_indices</code>.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    // Generate 4D Gauss-Legendre quadrature rules for various cell
    // neighboring types.
    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<4> quad_rule_for_regular(quad_order_for_regular);

    // Local matrix
    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    // Polynomial space inverse numbering for recovering the lexicographic
    // order.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in lexicographic order.
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            // Get permuted local DoF indices.
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_same_panel,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        case CommonEdge:
          {
            // This part handles the common edge case of Sauter's
            // quadrature rule.
            // 1. Get the DoF indices in lexicographic order for \f$K_x\f$.
            // 2. Get the DoF indices in reversed lexicographic order for
            // \f$K_x\f$.
            // 3. Extract DoF indices only for cell vertices in \f$K_x\f$ and
            // \f$K_y\f$. N.B. The DoF indices for the last two vertices are
            // swapped, such that the four vertices are in clockwise or
            // counter clockwise order.
            // 4. Determine the starting vertex.

            Assert(vertex_dof_index_intersection.size() ==
                     GeometryInfo<dim>::vertices_per_face,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            std::vector<unsigned int>
              ky_fe_reversed_poly_space_numbering_inverse =
                generate_backward_dof_permutation(ky_fe, 0);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_reversed_poly_space_numbering_inverse);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_backward_dof_permutation(ky_fe,
                                                ky_starting_vertex_index);

            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_common_edge,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        case CommonVertex:
          {
            Assert(vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            // Determine the starting vertex index in \f$K_x\f$ and \f$K_y\f$.
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            // Generate the permutation of DoFs in \f$K_x\f$ and \f$K_y\f$ by
            // starting from <code>kx_starting_vertex_index</code> or
            // <code>ky_starting_vertex_index</code>.
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_common_vertex,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        case Regular:
          {
            Assert(vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }


            // Iterate over DoFs for test function space in lexicographic
            // order in \f$K_x\f$.
            for (unsigned int i = 0; i < kx_n_dofs; i++)
              {
                // Iterate over DoFs for ansatz function space in tensor
                // product order in \f$K_y\f$.
                for (unsigned int j = 0; j < ky_n_dofs; j++)
                  {
                    // Pullback the kernel function to unit cell.
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(kernel_function,
                                              cell_neighboring_type,
                                              kx_support_points_permuted,
                                              ky_support_points_permuted,
                                              kx_fe,
                                              ky_fe,
                                              i,
                                              j);

                    // Pullback the kernel function to Sauter parameter
                    // space.
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type);

                    // Apply 4d Sauter numerical quadrature.
                    cell_matrix(i, j) =
                      ApplyQuadrature(quad_rule_for_regular,
                                      kernel_pullback_on_sauter);
                  }
              }

            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }

    // Assemble the cell matrix to system matrix.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            system_matrix.add(kx_local_dof_indices_permuted[i],
                              ky_local_dof_indices_permuted[j],
                              cell_matrix(i, j));
          }
      }
  }


  /**
   * This function implements Sauter's quadrature rule on quadrangular mesh.
   * It handles various cases including same panel, common edge, common vertex
   * and regular cell neighboring types.
   *
   * @param kernel_function Laplace kernel function.
   * @param bem_values
   * @param kx_cell_iter Iterator pointing to \f$K_x\f$.
   * @param kx_cell_iter Iterator pointing to \f$K_y\f$.
   * @param kx_mapping Mapping used for \f$K_x\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   * @param ky_mapping Mapping used for \f$K_y\f$. Because a mesher usually generates
   * 1st order grid, if there is no additional manifold specification, the
   * mapping should be 1st order.
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  SauterQuadRule(
    FullMatrix<RangeNumberType> &                            system_matrix,
    const KernelFunction<spacedim, RangeNumberType> &        kernel_function,
    const BEMValues<dim, spacedim> &                         bem_values,
    const typename DoFHandler<dim, spacedim>::cell_iterator &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::cell_iterator &ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &                   kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    std::vector<types::global_dof_index> vertex_dof_index_intersection;
    vertex_dof_index_intersection.reserve(vertices_per_cell);
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        vertex_dof_index_intersection);


    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    // Support points of \f$K_x\f$ and \f$K_y\f$ in the default
    // hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_hierarchical =
      get_hierarchic_support_points_in_real_cell(kx_cell_iter,
                                                 kx_fe,
                                                 kx_mapping);
    std::vector<Point<spacedim>> ky_support_points_hierarchical =
      get_hierarchic_support_points_in_real_cell(ky_cell_iter,
                                                 ky_fe,
                                                 ky_mapping);

    // Permuted support points to be used in the common edge and common vertex
    // cases instead of the original support points in hierarchical order.
    std::vector<Point<spacedim>> kx_support_points_permuted;
    std::vector<Point<spacedim>> ky_support_points_permuted;
    kx_support_points_permuted.reserve(kx_n_dofs);
    ky_support_points_permuted.reserve(ky_n_dofs);

    // Global indices for the local DoFs in the default hierarchical order.
    // N.B. These vectors should have the right size before being passed to
    // the function <code>get_dof_indices</code>.
    std::vector<types::global_dof_index> kx_local_dof_indices_hierarchical(
      kx_n_dofs);
    std::vector<types::global_dof_index> ky_local_dof_indices_hierarchical(
      ky_n_dofs);
    kx_cell_iter->get_dof_indices(kx_local_dof_indices_hierarchical);
    ky_cell_iter->get_dof_indices(ky_local_dof_indices_hierarchical);

    // Permuted local DoF indices, which has the same permutation as that
    // applied to support points.
    std::vector<types::global_dof_index> kx_local_dof_indices_permuted;
    std::vector<types::global_dof_index> ky_local_dof_indices_permuted;
    kx_local_dof_indices_permuted.reserve(kx_n_dofs);
    ky_local_dof_indices_permuted.reserve(ky_n_dofs);

    FullMatrix<RangeNumberType> cell_matrix(kx_n_dofs, ky_n_dofs);

    // Polynomial space inverse numbering for recovering the tensor
    // product ordering.
    std::vector<unsigned int> kx_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(kx_fe);
    std::vector<unsigned int> ky_fe_poly_space_numbering_inverse =
      FETools::lexicographic_to_hierarchic_numbering(ky_fe);

    // Quadrature rule to be adopted depending on the cell neighboring
    // type.
    const QGauss<4> *active_quad_rule = nullptr;

    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            Assert(vertex_dof_index_intersection.size() == vertices_per_cell,
                   ExcInternalError());

            // Get support points in the lexicographic order.
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            // Get permuted local DoF indices in the lexicographic order.
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            active_quad_rule = &(bem_values.quad_rule_for_same_panel);


            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        case CommonEdge:
          {
            /**
             * This part handles the common edge case of Sauter's quadrature
             * rule.
             *
             * 1. Get the DoF indices in the lexicographic order for \f$K_x\f$.
             * 2. Get the DoF indices in the reversed lexicographic order for
             * \f$K_y\f$. Hence, the orientation determined from these DoF
             * indices is opposite to that of \f$K_x\f$.
             * 3. Extract DoF indices only for cell vertices or corners in
             * \f$K_x\f$ and \f$K_y\f$.
             * \mynote{Because the four cell vertices or corners retrieved from
             * the list of DoFs either in the lexicographic or the reversed
             * lexicographic order are in the zigzag form as shown below,
             * @verbatim
             * 2 ----- 3
             * |  Kx   |
             * 0 ----- 1
             * |  Ky   |
             * 2 ----- 3
             * @endverbatim
             * the DoF indices for the last two vertices should be swapped, such
             * that the four vertices in the cell are either in the clockwise or
             * counter clockwise order, as shown below.
             * @verbatim
             * 3 ----- 2
             * |  Kx   |
             * 0 ----- 1
             * |  Ky   |
             * 3 ----- 2
             * @endverbatim
             * 4. Determine the starting vertex on the common edge.
             *
             * Finally, we should keep in mind that the orientation of the cell
             * \f$K_y\f$ is reversed due to the above operation, so the normal
             * vector \f$n_y\f$ calculated from such permuted support points and
             * shape functions should be negated back to the correct direction
             * during the evaluation of the kernel function.
             */
            Assert(vertex_dof_index_intersection.size() ==
                     GeometryInfo<dim>::vertices_per_face,
                   ExcInternalError());

            /**
             * Get permuted local DoF indices in \f$K_x\f$ in the lexicographic
             * order.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);
            /**
             * Get permuted local DoF indices in \f$K_y\f$ in the reversed
             * lexicographic order by starting from the first vertex.
             */
            std::vector<unsigned int>
              ky_fe_reversed_poly_space_numbering_inverse =
                generate_backward_dof_permutation(ky_fe, 0);
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_reversed_poly_space_numbering_inverse);

            /**
             * Get the DoF indices for the vertices or corners of \f$K_x\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);
            /**
             * Get the DoF indices for the vertices or corners of \f$K_y\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_x\f$.
             */
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());
            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_y\f$.
             */
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            /**
             * Generate the permutation of DoF indices in \f$K_x\f$ by starting
             * from the vertex <code>kx_starting_vertex_index</code> in the
             * lexicographic order, i.e. forward traversal.
             */
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);

            /**
             * Generate the permutation of DoF indices in \f$K_y\f$ by starting
             * from the vertex <code>ky_starting_vertex_index</code> in the
             * reversed lexicographic order, i.e. backward traversal.
             */
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_backward_dof_permutation(ky_fe,
                                                ky_starting_vertex_index);

            /**
             * Get the list of permuted support points in \f$K_x\f$ according
             * to the permutation @p kx_local_dof_permutation.
             */
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted support points in \f$K_y\f$ according
             * to the permutation @p ky_local_dof_permutation.
             */
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_x\f$ according to
             * the permutation @p kx_local_dof_permutation.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_y\f$ according to
             * the permutation @p ky_local_dof_permutation.
             */
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);

            active_quad_rule = &(bem_values.quad_rule_for_common_edge);

            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        case CommonVertex:
          {
            /**
             * This part handles the common vertex case of Sauter's quadrature
             * rule. This is simpler than that of the common edge case because
             * the orientations of \f$K_x\f$ and \f$K_y\f$ are intact.
             */
            Assert(vertex_dof_index_intersection.size() == 1,
                   ExcInternalError());

            /**
             * Get permuted local DoF indices in \f$K_x\f$ in the lexicographic
             * order.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            /**
             * Get permuted local DoF indices in \f$K_y\f$ in the lexicographic
             * order.
             */
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            /**
             * Get the DoF indices for the vertices or corners of \f$K_x\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              kx_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(kx_fe,
                                               kx_local_dof_indices_permuted);

            /**
             * Get the DoF indices for the vertices or corners of \f$K_y\f$
             * with the last two swapped.
             */
            std::array<types::global_dof_index, vertices_per_cell>
              ky_local_vertex_dof_indices_swapped =
                get_vertex_dof_indices_swapped(ky_fe,
                                               ky_local_dof_indices_permuted);

            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_x\f$.
             */
            unsigned int kx_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                kx_local_vertex_dof_indices_swapped);
            Assert(kx_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            /**
             * Determine the starting vertex index wrt. the list of vertex DoF
             * indices in \f$K_y\f$.
             */
            unsigned int ky_starting_vertex_index =
              get_start_vertex_dof_index<vertices_per_cell>(
                vertex_dof_index_intersection,
                ky_local_vertex_dof_indices_swapped);
            Assert(ky_starting_vertex_index < vertices_per_cell,
                   ExcInternalError());

            /**
             * Generate the permutation of DoF indices in \f$K_x\f$ by starting
             * from the vertex <code>kx_starting_vertex_index</code> in the
             * lexicographic order, i.e. forward traversal.
             */
            std::vector<unsigned int> kx_local_dof_permutation =
              generate_forward_dof_permutation(kx_fe, kx_starting_vertex_index);

            /**
             * Generate the permutation of DoF indices in \f$K_y\f$ by starting
             * from the vertex <code>ky_starting_vertex_index</code> in the
             * lexicographic order, i.e. forward traversal.
             */
            std::vector<unsigned int> ky_local_dof_permutation =
              generate_forward_dof_permutation(ky_fe, ky_starting_vertex_index);

            /**
             * Get the list of permuted support points in \f$K_x\f$ according
             * to the permutation @p kx_local_dof_permutation.
             */
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted support points in \f$K_y\f$ according
             * to the permutation @p ky_local_dof_permutation.
             */
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_x\f$ according to
             * the permutation @p kx_local_dof_permutation.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_local_dof_permutation);

            /**
             * Get the list of permuted DoF indices in \f$K_y\f$ according to
             * the permutation @p ky_local_dof_permutation.
             */
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_local_dof_permutation);

            active_quad_rule = &(bem_values.quad_rule_for_common_vertex);

            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        case Regular:
          {
            /**
             * This part handles the regular case of Sauter's quadrature rule.
             */
            Assert(vertex_dof_index_intersection.size() == 0,
                   ExcInternalError());

            /**
             * Get permuted local DoF indices in \f$K_x\f$ in the lexicographic
             * order.
             */
            kx_local_dof_indices_permuted =
              permute_vector(kx_local_dof_indices_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            /**
             * Get permuted local DoF indices in \f$K_y\f$ in the lexicographic
             * order.
             */
            ky_local_dof_indices_permuted =
              permute_vector(ky_local_dof_indices_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            /**
             * Get the list of permuted support points in \f$K_x\f$ in the
             * lexicographic order.
             */
            kx_support_points_permuted =
              permute_vector(kx_support_points_hierarchical,
                             kx_fe_poly_space_numbering_inverse);

            /**
             * Get the list of permuted support points in \f$K_y\f$ in the
             * lexicographic order.
             */
            ky_support_points_permuted =
              permute_vector(ky_support_points_hierarchical,
                             ky_fe_poly_space_numbering_inverse);

            active_quad_rule = &(bem_values.quad_rule_for_regular);

            //                // DEBUG: Print out permuted support points
            //                and DoF indices for
            //                // debugging.
            //                deallog << "Support points and DoF indices
            //                in Kx:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < kx_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    kx_local_dof_indices_permuted[i] << " "
            //                            << kx_support_points_permuted[i]
            //                            << std::endl;
            //                  }
            //
            //                deallog << "Support points and DoF indices
            //                in Ky:\n"; deallog << "DoF_index X Y Z\n";
            //                for (unsigned int i = 0; i < ky_n_dofs; i++)
            //                  {
            //                    deallog <<
            //                    ky_local_dof_indices_permuted[i] << " "
            //                            << ky_support_points_permuted[i]
            //                            << std::endl;
            //                  }

            break;
          }
        default:
          {
            Assert(false, ExcNotImplemented());
            active_quad_rule = nullptr;
          }
      }

    /**
     * Iterate over DoFs for test function space in \f$K_x\f$.
     */
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        /**
         * Iterate over DoFs for ansatz function space in \f$K_y\f$.
         */
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            /**
             * Pullback the kernel function to unit cell.
             */
            KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
              kernel_pullback_on_unit(kernel_function,
                                      cell_neighboring_type,
                                      kx_support_points_permuted,
                                      ky_support_points_permuted,
                                      kx_fe,
                                      ky_fe,
                                      &bem_values,
                                      i,
                                      j);

            /**
             * Pullback the kernel function to Sauter parameter space.
             */
            KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>
              kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                        cell_neighboring_type,
                                        &bem_values);

            /**
             * Apply 4d Sauter numerical quadrature.
             */
            cell_matrix(i, j) =
              ApplyQuadratureUsingBEMValues(*active_quad_rule,
                                            kernel_pullback_on_sauter);
          }
      }

    /**
     * Assemble the cell matrix to system matrix.
     */
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            system_matrix.add(kx_local_dof_indices_permuted[i],
                              ky_local_dof_indices_permuted[j],
                              cell_matrix(i, j));
          }
      }
  }


  /**
   * Apply the Sauter's 4D quadrature rule to the kernel function pulled back to
   * the Sauter's parametric space.
   *
   * @param quad_rule
   * @param f
   * @param component
   * @return
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  RangeNumberType
  ApplyQuadrature(
    const Quadrature<dim * 2> &quad_rule,
    const KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType> &f,
    unsigned int component = 0)
  {
    RangeNumberType result = 0.;

    const std::vector<Point<dim * 2>> &quad_points  = quad_rule.get_points();
    const std::vector<double> &        quad_weights = quad_rule.get_weights();

    for (unsigned int q = 0; q < quad_rule.size(); q++)
      {
        result += f.value(quad_points[q], component) * quad_weights[q];
      }

    return result;
  }


  /**
   * Apply the Sauter's 4D quadrature rule to the kernel function pulled back to
   * the Sauter's parametric space. This version uses the precalculated
   * @p BEMValues.
   *
   * @param quad_rule
   * @param f
   * @param component
   * @return
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  RangeNumberType
  ApplyQuadratureUsingBEMValues(
    const Quadrature<dim * 2> &quad_rule,
    const KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType> &f,
    unsigned int component = 0)
  {
    RangeNumberType result = 0.;

    const std::vector<double> &quad_weights = quad_rule.get_weights();

    for (unsigned int q = 0; q < quad_rule.size(); q++)
      {
        // Evaluate the integrand with precalculated shape values and shape
        // gradient matrices.
        result += f.value(q, component) * quad_weights[q];
      }

    return result;
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  const QGauss<dim * 2> &
  select_sauter_quad_rule_from_bem_values(
    const CellNeighboringType                        cell_neighboring_type,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values)
  {
    switch (cell_neighboring_type)
      {
        case SamePanel:
          {
            return bem_values.quad_rule_for_same_panel;
          }
        case CommonEdge:
          {
            return bem_values.quad_rule_for_common_edge;
          }
        case CommonVertex:
          {
            return bem_values.quad_rule_for_common_vertex;
          }
        case Regular:
          {
            return bem_values.quad_rule_for_regular;
          }
        default:
          {
            Assert(false, ExcNotImplemented());

            return bem_values.quad_rule_for_same_panel;
          }
      }
  }


  /**
   * Perform the Galerkin-BEM double integral using Sauter's quadrature.
   *
   * \mynote{1. This version is used for parallelization, where the computation
   * results will be used to fill the @p scratch and @p data.
   * 2. At present, both SLP and DLP kernels are computed.}
   *
   * @param scratch
   * @param data
   * @param slp
   * @param dlp
   * @param bem_values
   * @param kx_cell_iter
   * @param ky_cell_iter
   * @param kx_mapping
   * @param ky_mapping
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  sauter_assemble_on_one_pair_of_cells(
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 slp,
    const KernelFunction<spacedim> &                 dlp,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &                                   ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    scratch.vertex_dof_index_intersection.clear();
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        scratch.vertex_dof_index_intersection);
    // Quadrature rule to be adopted depending on the cell neighboring
    // type.
    const QGauss<dim * 2> active_quad_rule =
      select_sauter_quad_rule_from_bem_values(cell_neighboring_type,
                                              bem_values);

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    permute_dofs_for_sauter_quad(scratch,
                                 data,
                                 cell_neighboring_type,
                                 kx_cell_iter,
                                 ky_cell_iter,
                                 kx_mapping,
                                 ky_mapping);

    calc_jacobian_normals_for_sauter_quad(scratch,
                                          cell_neighboring_type,
                                          bem_values,
                                          active_quad_rule);

    /**
     *  Clear the local matrix in case that it is reused from another
     *  finished task. N.B. Its memory has already been allocated in the
     *  constructor of @p CellPairWisePerTaskData.
     */
    data.dlp_matrix = 0.;
    data.slp_matrix = 0.;

    // Iterate over DoFs for test function space in \f$K_x\f$.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        // Iterate over DoFs for ansatz function space in \f$K_y\f$.
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            // Pullback the DLP kernel function to unit cell.
            KernelPulledbackToUnitCell<dim, spacedim, double>
              dlp_kernel_pullback_on_unit(dlp,
                                          cell_neighboring_type,
                                          scratch.kx_support_points_permuted,
                                          scratch.ky_support_points_permuted,
                                          kx_fe,
                                          ky_fe,
                                          &bem_values,
                                          &scratch,
                                          i,
                                          j);

            // Pullback the DLP kernel function to Sauter parameter
            // space.
            KernelPulledbackToSauterSpace<dim, spacedim, double>
              dlp_kernel_pullback_on_sauter(dlp_kernel_pullback_on_unit,
                                            cell_neighboring_type,
                                            &bem_values);

            // Apply 4d Sauter numerical quadrature.
            data.dlp_matrix(i, j) =
              ApplyQuadratureUsingBEMValues(active_quad_rule,
                                            dlp_kernel_pullback_on_sauter);

            // Pullback the SLP kernel function to unit cell.
            KernelPulledbackToUnitCell<dim, spacedim, double>
              slp_kernel_pullback_on_unit(slp,
                                          cell_neighboring_type,
                                          scratch.kx_support_points_permuted,
                                          scratch.ky_support_points_permuted,
                                          kx_fe,
                                          ky_fe,
                                          &bem_values,
                                          &scratch,
                                          i,
                                          j);

            // Pullback the SLP kernel function to Sauter parameter
            // space.
            KernelPulledbackToSauterSpace<dim, spacedim, double>
              slp_kernel_pullback_on_sauter(slp_kernel_pullback_on_unit,
                                            cell_neighboring_type,
                                            &bem_values);

            // Apply 4d Sauter numerical quadrature.
            data.slp_matrix(i, j) =
              ApplyQuadratureUsingBEMValues(active_quad_rule,
                                            slp_kernel_pullback_on_sauter);
          }
      }
  }


  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  sauter_assemble_on_one_pair_of_cells(
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 slp,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &kx_cell_iter,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator
      &                                   ky_cell_iter,
    const MappingQGeneric<dim, spacedim> &kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    // Determine the cell neighboring type based on the vertex dof indices.
    // The common dof indices will be stored into the vector
    // <code>vertex_dof_index_intersection</code> if there is any.
    std::array<types::global_dof_index, vertices_per_cell>
      kx_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
    std::array<types::global_dof_index, vertices_per_cell>
      ky_vertex_dof_indices(
        get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

    scratch.vertex_dof_index_intersection.clear();
    CellNeighboringType cell_neighboring_type =
      detect_cell_neighboring_type<dim>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        scratch.vertex_dof_index_intersection);
    // Quadrature rule to be adopted depending on the cell neighboring
    // type.
    const QGauss<dim * 2> active_quad_rule =
      select_sauter_quad_rule_from_bem_values(cell_neighboring_type,
                                              bem_values);

    const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
    const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

    const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
    const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

    permute_dofs_for_sauter_quad(scratch,
                                 data,
                                 cell_neighboring_type,
                                 kx_cell_iter,
                                 ky_cell_iter,
                                 kx_mapping,
                                 ky_mapping);

    calc_jacobian_normals_for_sauter_quad(scratch,
                                          cell_neighboring_type,
                                          bem_values,
                                          active_quad_rule);

    /**
     *  Clear the local matrix in case that it is reused from another
     *  finished task. N.B. Its memory has already been allocated in the
     *  constructor of @p CellPairWisePerTaskData.
     */
    data.dlp_matrix = 0.;

    // Iterate over DoFs for test function space in \f$K_x\f$.
    for (unsigned int i = 0; i < kx_n_dofs; i++)
      {
        // Iterate over DoFs for ansatz function space in \f$K_y\f$.
        for (unsigned int j = 0; j < ky_n_dofs; j++)
          {
            // Pullback the SLP kernel function to unit cell.
            KernelPulledbackToUnitCell<dim, spacedim, double>
              slp_kernel_pullback_on_unit(slp,
                                          cell_neighboring_type,
                                          scratch.kx_support_points_permuted,
                                          scratch.ky_support_points_permuted,
                                          kx_fe,
                                          ky_fe,
                                          &bem_values,
                                          &scratch,
                                          i,
                                          j);

            // Pullback the SLP kernel function to Sauter parameter
            // space.
            KernelPulledbackToSauterSpace<dim, spacedim, double>
              slp_kernel_pullback_on_sauter(slp_kernel_pullback_on_unit,
                                            cell_neighboring_type,
                                            &bem_values);

            // Apply 4d Sauter numerical quadrature.
            data.slp_matrix(i, j) =
              ApplyQuadratureUsingBEMValues(active_quad_rule,
                                            slp_kernel_pullback_on_sauter);
          }
      }
  }


  /**
   * Perform Galerkin-BEM double integral with respect to a given kernel on a
   * pair of DoFs \f$(i, j)\f$ using the Sauter quadrature.
   *
   * Assume \f$\mathcal{K}_i\f$ is the collection of cells sharing the DoF
   * support point \f$i\f$ and \f$\mathcal{K}_j\f$ is the collection of cells
   * sharing the DoF support point \f$j\f$. Then Galerkin-BEM double integral
   * will be over each cell pair which is comprised of an arbitrary cell in
   * \f$\mathcal{K}_i\f$ and an arbitrary cell in \f$\mathcal{K}_j\f$.
   *
   * @param scratch
   * @param data
   * @param kernel
   * @param i
   * @param j
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   * @return
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  RangeNumberType
  sauter_assemble_on_one_pair_of_dofs(
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const KernelFunction<spacedim> &                 kernel,
    const types::global_dof_index                    i,
    const types::global_dof_index                    j,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    RangeNumberType double_integral = 0.0;

    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    /**
     * Iterate over each cell in the support of the basis function for the i-th
     * DoF.
     */
    for (unsigned int kx_cell_index : dof_to_cell_topo[i])
      {
        typename DoFHandler<dim, spacedim>::active_cell_iterator kx_cell_iter =
          kx_dof_handler.active_cell_iterators().begin();
        std::advance(kx_cell_iter, kx_cell_index);
        /**
         * Iterate over each cell in the support of the basis function for the
         * j-th DoF.
         */
        for (unsigned int ky_cell_index : dof_to_cell_topo[j])
          {
            typename DoFHandler<dim, spacedim>::active_cell_iterator
              ky_cell_iter = ky_dof_handler.active_cell_iterators().begin();
            std::advance(ky_cell_iter, ky_cell_index);

            // Determine the cell neighboring type based on the vertex dof
            // indices. The common dof indices will be stored into the vector
            // <code>vertex_dof_index_intersection</code> if there is any.
            std::array<types::global_dof_index, vertices_per_cell>
              kx_vertex_dof_indices(
                get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
            std::array<types::global_dof_index, vertices_per_cell>
              ky_vertex_dof_indices(
                get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

            scratch.vertex_dof_index_intersection.clear();
            CellNeighboringType cell_neighboring_type =
              detect_cell_neighboring_type<dim>(
                kx_vertex_dof_indices,
                ky_vertex_dof_indices,
                scratch.vertex_dof_index_intersection);
            // Quadrature rule to be adopted depending on the cell neighboring
            // type.
            const QGauss<dim * 2> active_quad_rule =
              select_sauter_quad_rule_from_bem_values(cell_neighboring_type,
                                                      bem_values);

            const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
            const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

            permute_dofs_for_sauter_quad(scratch,
                                         data,
                                         cell_neighboring_type,
                                         kx_cell_iter,
                                         ky_cell_iter,
                                         kx_mapping,
                                         ky_mapping);

            calc_jacobian_normals_for_sauter_quad(scratch,
                                                  cell_neighboring_type,
                                                  bem_values,
                                                  active_quad_rule);

            /**
             * Find the index of the i-th DoF in the permuted DoF indices of
             * \f$K_x\f$.
             */
            typename std::vector<types::global_dof_index>::const_iterator
              i_iter = std::find(data.kx_local_dof_indices_permuted.begin(),
                                 data.kx_local_dof_indices_permuted.end(),
                                 i);
            Assert(i_iter != data.kx_local_dof_indices_permuted.end(),
                   ExcInternalError());
            unsigned int i_index =
              i_iter - data.kx_local_dof_indices_permuted.begin();

            /**
             * Find the index of the j-th DoF in the permuted DoF indices of
             * \f$K_y\f$.
             */
            typename std::vector<types::global_dof_index>::const_iterator
              j_iter = std::find(data.ky_local_dof_indices_permuted.begin(),
                                 data.ky_local_dof_indices_permuted.end(),
                                 j);
            Assert(j_iter != data.ky_local_dof_indices_permuted.end(),
                   ExcInternalError());
            unsigned int j_index =
              j_iter - data.ky_local_dof_indices_permuted.begin();


            /**
             * Pullback the kernel function to unit cell.
             */
            KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
              kernel_pullback_on_unit(kernel,
                                      cell_neighboring_type,
                                      scratch.kx_support_points_permuted,
                                      scratch.ky_support_points_permuted,
                                      kx_fe,
                                      ky_fe,
                                      &bem_values,
                                      &scratch,
                                      i_index,
                                      j_index);

            /**
             * Pullback the kernel function to Sauter parameter space.
             */
            KernelPulledbackToSauterSpace<dim, spacedim, RangeNumberType>
              kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                        cell_neighboring_type,
                                        &bem_values);

            // Apply 4d Sauter numerical quadrature.
            double_integral +=
              ApplyQuadratureUsingBEMValues(active_quad_rule,
                                            kernel_pullback_on_sauter);
          }
      }

    return double_integral;
  }


  /**
   * Perform Galerkin-BEM double integral with respect to a list of kernels on
   * a pair of DoFs \f$(i, j)\f$ using the Sauter quadrature.
   *
   * Assume \f$\mathcal{K}_i\f$ is the collection of cells sharing the DoF
   * support point \f$i\f$ and \f$\mathcal{K}_j\f$ is the collection of cells
   * sharing the DoF support point \f$j\f$. Then Galerkin-BEM double integral
   * will be over each cell pair which is comprised of an arbitrary cell in
   * \f$\mathcal{K}_i\f$ and an arbitrary cell in \f$\mathcal{K}_j\f$.
   *
   * @param results The vector of values returned for the integral with respect
   * to the vector of kernel functions
   * @param scratch
   * @param data
   * @param kernels A vector of kernel function pointers
   * @param enable_kernel_evaluations A list of flags indicating if each kernel
   * is to be evaluated.
   * @param i
   * @param j
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  sauter_assemble_on_one_pair_of_dofs(
    Vector<RangeNumberType> &                        results,
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<bool> &                        enable_kernel_evaluations,
    const types::global_dof_index                    i,
    const types::global_dof_index                    j,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    /**
     * Reinitialize the result vector to zero.
     */
    results.reinit(kernels.size());

    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    /**
     * Iterate over each cell in the support of the basis function for the i-th
     * DoF.
     */
    for (unsigned int kx_cell_index : dof_to_cell_topo[i])
      {
        typename DoFHandler<dim, spacedim>::active_cell_iterator kx_cell_iter =
          kx_dof_handler.active_cell_iterators().begin();
        std::advance(kx_cell_iter, kx_cell_index);
        /**
         * Iterate over each cell in the support of the basis function for the
         * j-th DoF.
         */
        for (unsigned int ky_cell_index : dof_to_cell_topo[j])
          {
            typename DoFHandler<dim, spacedim>::active_cell_iterator
              ky_cell_iter = ky_dof_handler.active_cell_iterators().begin();
            std::advance(ky_cell_iter, ky_cell_index);

            // Determine the cell neighboring type based on the vertex dof
            // indices. The common dof indices will be stored into the vector
            // <code>vertex_dof_index_intersection</code> if there is any.
            std::array<types::global_dof_index, vertices_per_cell>
              kx_vertex_dof_indices(
                get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
            std::array<types::global_dof_index, vertices_per_cell>
              ky_vertex_dof_indices(
                get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

            scratch.vertex_dof_index_intersection.clear();
            CellNeighboringType cell_neighboring_type =
              detect_cell_neighboring_type<dim>(
                kx_vertex_dof_indices,
                ky_vertex_dof_indices,
                scratch.vertex_dof_index_intersection);
            // Quadrature rule to be adopted depending on the cell neighboring
            // type.
            const QGauss<dim * 2> active_quad_rule =
              select_sauter_quad_rule_from_bem_values(cell_neighboring_type,
                                                      bem_values);

            const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
            const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

            permute_dofs_for_sauter_quad(scratch,
                                         data,
                                         cell_neighboring_type,
                                         kx_cell_iter,
                                         ky_cell_iter,
                                         kx_mapping,
                                         ky_mapping);

            calc_jacobian_normals_for_sauter_quad(scratch,
                                                  cell_neighboring_type,
                                                  bem_values,
                                                  active_quad_rule);

            /**
             * Find the index of the i-th DoF in the permuted DoF indices of
             * \f$K_x\f$.
             */
            typename std::vector<types::global_dof_index>::const_iterator
              i_iter = std::find(data.kx_local_dof_indices_permuted.begin(),
                                 data.kx_local_dof_indices_permuted.end(),
                                 i);
            Assert(i_iter != data.kx_local_dof_indices_permuted.end(),
                   ExcInternalError());
            unsigned int i_index =
              i_iter - data.kx_local_dof_indices_permuted.begin();

            /**
             * Find the index of the j-th DoF in the permuted DoF indices of
             * \f$K_y\f$.
             */
            typename std::vector<types::global_dof_index>::const_iterator
              j_iter = std::find(data.ky_local_dof_indices_permuted.begin(),
                                 data.ky_local_dof_indices_permuted.end(),
                                 j);
            Assert(j_iter != data.ky_local_dof_indices_permuted.end(),
                   ExcInternalError());
            unsigned int j_index =
              j_iter - data.ky_local_dof_indices_permuted.begin();


            /**
             * Iterate over each kernel.
             */
            unsigned int counter = 0;
            for (const KernelFunction<spacedim> *kernel : kernels)
              {
                if (enable_kernel_evaluations[counter])
                  {
                    /**
                     * Pullback the kernel function to unit cell.
                     */
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(
                        *kernel,
                        cell_neighboring_type,
                        scratch.kx_support_points_permuted,
                        scratch.ky_support_points_permuted,
                        kx_fe,
                        ky_fe,
                        &bem_values,
                        &scratch,
                        i_index,
                        j_index);

                    /**
                     * Pullback the kernel function to Sauter parameter space.
                     */
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type,
                                                &bem_values);

                    /**
                     * Apply 4d Sauter numerical quadrature.
                     */
                    results(counter) +=
                      ApplyQuadratureUsingBEMValues(active_quad_rule,
                                                    kernel_pullback_on_sauter);
                  }

                counter++;
              }
          }
      }
  }


  /**
   * Perform Galerkin-BEM double integral with respect to a list of kernels on
   * a pair of DoFs \f$(i, j)\f$ using the Sauter quadrature. In the meantime,
   * if these two DoFs are shared by some common cells, the FEM mass matrix
   * multiplied by a factor will be added into the result.
   *
   * Assume \f$\mathcal{K}_i\f$ is the collection of cells sharing the DoF
   * support point \f$i\f$ and \f$\mathcal{K}_j\f$ is the collection of cells
   * sharing the DoF support point \f$j\f$. Then Galerkin-BEM double integral
   * will be over each cell pair which is comprised of an arbitrary cell in
   * \f$\mathcal{K}_i\f$ and an arbitrary cell in \f$\mathcal{K}_j\f$.
   *
   * @param results
   * @param scratch
   * @param data
   * @param mass_matrix_factors
   * @param kernels A vector of kernel function pointers
   * @param enable_kernel_evaluations A list of flags indicating if each kernel
   * is to be evaluated.
   * @param i
   * @param j
   * @param dof_to_cell_topo
   * @param bem_values
   * @param kx_dof_handler
   * @param ky_dof_handler
   * @param kx_mapping
   * @param ky_mapping
   */
  template <int dim, int spacedim, typename RangeNumberType = double>
  void
  sauter_assemble_on_one_pair_of_dofs(
    Vector<RangeNumberType> &                        results,
    PairCellWiseScratchData &                        scratch,
    PairCellWisePerTaskData &                        data,
    CellWiseScratchData &                            fem_scratch,
    const std::vector<RangeNumberType> &             mass_matrix_factors,
    const std::vector<KernelFunction<spacedim> *> &  kernels,
    const std::vector<bool> &                        enable_kernel_evaluations,
    const types::global_dof_index                    i,
    const types::global_dof_index                    j,
    const std::vector<std::vector<unsigned int>> &   dof_to_cell_topo,
    const BEMValues<dim, spacedim, RangeNumberType> &bem_values,
    const DoFHandler<dim, spacedim> &                kx_dof_handler,
    const DoFHandler<dim, spacedim> &                ky_dof_handler,
    const MappingQGeneric<dim, spacedim> &           kx_mapping =
      MappingQGeneric<dim, spacedim>(1),
    const MappingQGeneric<dim, spacedim> &ky_mapping =
      MappingQGeneric<dim, spacedim>(1))
  {
    /**
     * Reinitialize the result vector to zero.
     */
    results.reinit(kernels.size());

    // Geometry information.
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;

    /**
     * Iterate over each cell in the support of the basis function for the i-th
     * DoF.
     */
    for (unsigned int kx_cell_index : dof_to_cell_topo[i])
      {
        /**
         * When the FEM mass matrix is to be computed and appended, this
         * indicates whether the @p FEValues for the current cell \f$K_x\f$
         * should be recalculated. When we come to a new \f$K_x\f$, this flag
         * is set to true.
         */
        bool is_update_kx_fe_values = true;

        typename DoFHandler<dim, spacedim>::active_cell_iterator kx_cell_iter =
          kx_dof_handler.active_cell_iterators().begin();
        std::advance(kx_cell_iter, kx_cell_index);
        /**
         * Iterate over each cell in the support of the basis function for the
         * j-th DoF.
         */
        for (unsigned int ky_cell_index : dof_to_cell_topo[j])
          {
            typename DoFHandler<dim, spacedim>::active_cell_iterator
              ky_cell_iter = ky_dof_handler.active_cell_iterators().begin();
            std::advance(ky_cell_iter, ky_cell_index);

            // Determine the cell neighboring type based on the vertex dof
            // indices. The common dof indices will be stored into the vector
            // <code>vertex_dof_index_intersection</code> if there is any.
            std::array<types::global_dof_index, vertices_per_cell>
              kx_vertex_dof_indices(
                get_vertex_dof_indices<dim, spacedim>(kx_cell_iter));
            std::array<types::global_dof_index, vertices_per_cell>
              ky_vertex_dof_indices(
                get_vertex_dof_indices<dim, spacedim>(ky_cell_iter));

            scratch.vertex_dof_index_intersection.clear();
            CellNeighboringType cell_neighboring_type =
              detect_cell_neighboring_type<dim>(
                kx_vertex_dof_indices,
                ky_vertex_dof_indices,
                scratch.vertex_dof_index_intersection);
            // Quadrature rule to be adopted depending on the cell neighboring
            // type.
            const QGauss<dim * 2> active_quad_rule =
              select_sauter_quad_rule_from_bem_values(cell_neighboring_type,
                                                      bem_values);

            const FiniteElement<dim, spacedim> &kx_fe = kx_cell_iter->get_fe();
            const FiniteElement<dim, spacedim> &ky_fe = ky_cell_iter->get_fe();

            permute_dofs_for_sauter_quad(scratch,
                                         data,
                                         cell_neighboring_type,
                                         kx_cell_iter,
                                         ky_cell_iter,
                                         kx_mapping,
                                         ky_mapping);

            calc_jacobian_normals_for_sauter_quad(scratch,
                                                  cell_neighboring_type,
                                                  bem_values,
                                                  active_quad_rule);

            /**
             * Find the index of the i-th DoF in the permuted DoF indices of
             * \f$K_x\f$.
             */
            typename std::vector<types::global_dof_index>::const_iterator
              i_iter = std::find(data.kx_local_dof_indices_permuted.begin(),
                                 data.kx_local_dof_indices_permuted.end(),
                                 i);
            Assert(i_iter != data.kx_local_dof_indices_permuted.end(),
                   ExcInternalError());
            unsigned int i_index =
              i_iter - data.kx_local_dof_indices_permuted.begin();

            /**
             * Find the index of the j-th DoF in the permuted DoF indices of
             * \f$K_y\f$.
             */
            typename std::vector<types::global_dof_index>::const_iterator
              j_iter = std::find(data.ky_local_dof_indices_permuted.begin(),
                                 data.ky_local_dof_indices_permuted.end(),
                                 j);
            Assert(j_iter != data.ky_local_dof_indices_permuted.end(),
                   ExcInternalError());
            unsigned int j_index =
              j_iter - data.ky_local_dof_indices_permuted.begin();


            /**
             * Iterate over each kernel.
             */
            unsigned int counter = 0;
            for (const KernelFunction<spacedim> *kernel : kernels)
              {
                if (enable_kernel_evaluations[counter])
                  {
                    /**
                     * Pullback the kernel function to unit cell.
                     */
                    KernelPulledbackToUnitCell<dim, spacedim, RangeNumberType>
                      kernel_pullback_on_unit(
                        *kernel,
                        cell_neighboring_type,
                        scratch.kx_support_points_permuted,
                        scratch.ky_support_points_permuted,
                        kx_fe,
                        ky_fe,
                        &bem_values,
                        &scratch,
                        i_index,
                        j_index);

                    /**
                     * Pullback the kernel function to Sauter parameter space.
                     */
                    KernelPulledbackToSauterSpace<dim,
                                                  spacedim,
                                                  RangeNumberType>
                      kernel_pullback_on_sauter(kernel_pullback_on_unit,
                                                cell_neighboring_type,
                                                &bem_values);

                    /**
                     * Apply 4d Sauter numerical quadrature.
                     */
                    results(counter) +=
                      ApplyQuadratureUsingBEMValues(active_quad_rule,
                                                    kernel_pullback_on_sauter);

                    /**
                     * Append the FEM mass matrix contribution.
                     */
                    if ((kx_cell_index == ky_cell_index) &&
                        (mass_matrix_factors[counter] != 0))
                      {
                        if (is_update_kx_fe_values)
                          {
                            fem_scratch.fe_values.reinit(kx_cell_iter);
                            is_update_kx_fe_values = false;
                          }

                        const unsigned int n_q_points =
                          fem_scratch.fe_values.get_quadrature().size();

                        /**
                         * Get the index of the global DoF index \f$i\f$ in the
                         * current cell \f$K_x\f$.
                         */
                        auto i_local_dof_iter = std::find(
                          scratch.kx_local_dof_indices_hierarchical.begin(),
                          scratch.kx_local_dof_indices_hierarchical.end(),
                          i);
                        Assert(
                          i_local_dof_iter !=
                            scratch.kx_local_dof_indices_hierarchical.end(),
                          ExcMessage(
                            std::string("Cannot find the global DoF index ") +
                            std::to_string(i) +
                            std::string(" in the list of cell DoF indices!")));
                        const unsigned int i_local_dof_index =
                          i_local_dof_iter -
                          scratch.kx_local_dof_indices_hierarchical.begin();

                        /**
                         * Get the index of the global DoF index \f$j\f$ in the
                         * current cell \f$K_x\f$.
                         */
                        auto j_local_dof_iter = std::find(
                          scratch.kx_local_dof_indices_hierarchical.begin(),
                          scratch.kx_local_dof_indices_hierarchical.end(),
                          j);
                        Assert(
                          j_local_dof_iter !=
                            scratch.kx_local_dof_indices_hierarchical.end(),
                          ExcMessage(
                            std::string("Cannot find the global DoF index ") +
                            std::to_string(j) +
                            std::string(" in the list of cell DoF indices!")));
                        const unsigned int j_local_dof_index =
                          j_local_dof_iter -
                          scratch.kx_local_dof_indices_hierarchical.begin();

                        for (unsigned int q = 0; q < n_q_points; q++)
                          {
                            results(counter) +=
                              mass_matrix_factors[counter] *
                              fem_scratch.fe_values.shape_value(
                                i_local_dof_index, q) *
                              fem_scratch.fe_values.shape_value(
                                j_local_dof_index, q) *
                              fem_scratch.fe_values.JxW(q);
                          }
                      }
                  }

                counter++;
              }
          }
      }
  }
} // namespace IdeoBEM

#endif /* INCLUDE_SAUTER_QUADRATURE_H_ */
