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
 * @file bem_function_space.h
 * @brief Definition of a class for BEM function space.
 *
 * @date 2025-11-14
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_BEM_BEM_FUNCTION_SPACE_H_
#define HIERBEM_INCLUDE_BEM_BEM_FUNCTION_SPACE_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <memory>
#include <vector>

#include "bem_tools.h"
#include "cluster_tree/cluster_tree.h"
#include "cluster_tree/cluster_tree_builder.h"
#include "config.h"
#include "dofs/dof_to_cell_topology.h"
#include "dofs/dof_tools_ext.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * @brief Class for BEM function space.
 * @tparam Number Point coordinate number type
 * @tparam dim
 * @tparam spacedim
 * @tparam SearchableMaterialIdContainer
 */
template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number = double>
class BEMFunctionSpace
{
public:
  /**
   * @brief Construct a function space on the whole domain.
   * @param dof_handler_
   * @param n_min
   * @param cutoff_level During building the cluster tree associated with this
   * function space, when the level of a cluster is smaller then this level, the
   * partition from this cluster will be sent to a TBB task.
   */
  BEMFunctionSpace(const DoFHandler<dim, spacedim> &dof_handler_,
                   const unsigned int               n_min,
                   const unsigned int               cutoff_level = 0);

  /**
   * Construct a function space on a material subdomain.
   */
  BEMFunctionSpace(const DoFHandler<dim, spacedim>     &dof_handler_,
                   const SearchableMaterialIdContainer &material_ids_,
                   const unsigned int                   n_min,
                   const unsigned int                   cutoff_level = 0,
                   const bool include_boundary_dofs_                 = true,
                   const bool limit_support_in_subdomain_            = false);

  /**
   * Convert a vector from internal numbering to external numbering.
   */
  template <typename VectorType>
  void
  convert_vector_i2e(const VectorType &vec_i, VectorType &vec_e) const;

  /**
   * Convert a vector from external numbering to internal numbering.
   */
  template <typename VectorType>
  void
  convert_vector_e2i(const VectorType &vec_e, VectorType &vec_i) const;

  /**
   * Extract selected DoFs from a vector of all DoF values.
   *
   * @param vec Vector with elements for all DoFs in the DoF handler, using the
   * external numbering.
   * @param subvec Vector with elements for selected DoFs only, using the
   * external numbering.
   */
  template <typename VectorType>
  void
  extract_selected_dofs(const VectorType &vec, VectorType &subvec) const;

  /**
   * Assembel the vector of selected DoFs into a vector of all DoF values.
   */
  template <typename VectorType>
  void
  assemble_selected_dofs_into_full_dofs(const VectorType &subvec,
                                        VectorType       &vec) const;

  DoFHandler<dim, spacedim> &
  get_dof_handler()
  {
    return dof_handler;
  }

  const DoFHandler<dim, spacedim> &
  get_dof_handler() const
  {
    return dof_handler;
  }

  bool
  get_is_full_domain() const
  {
    return is_full_domain;
  }

  std::vector<bool> &
  get_dof_selectors()
  {
    return dof_selectors;
  }

  const std::vector<bool> &
  get_dof_selectors() const
  {
    return dof_selectors;
  }

  ClusterTree<spacedim, Number> &
  get_cluster_tree()
  {
    return *cluster_tree;
  }

  const ClusterTree<spacedim, Number> &
  get_cluster_tree() const
  {
    return *cluster_tree;
  }

  ClusterTreeBuilder<spacedim, Number> &
  get_cluster_tree_builder()
  {
    return *cluster_tree_builder;
  }

  const ClusterTreeBuilder<spacedim, Number> &
  get_cluster_tree_builder() const
  {
    return *cluster_tree_builder;
  }

  std::vector<Point<spacedim, Number>> &
  get_support_points()
  {
    return cluster_tree_builder->get_support_points();
  }

  const std::vector<Point<spacedim, Number>> &
  get_support_points() const
  {
    return cluster_tree_builder->get_support_points();
  }

  std::vector<Number> &
  get_dof_support_set_diameters()
  {
    return cluster_tree_builder->get_dof_support_set_diameters();
  }

  const std::vector<Number> &
  get_dof_support_set_diameters() const
  {
    return cluster_tree_builder->get_dof_support_set_diameters();
  }

  std::vector<types::global_dof_index> &
  get_internal_to_external_dof_numbering()
  {
    return cluster_tree->get_internal_to_external_dof_numbering();
  }

  const std::vector<types::global_dof_index> &
  get_internal_to_external_dof_numbering() const
  {
    return cluster_tree->get_internal_to_external_dof_numbering();
  }

  std::vector<types::global_dof_index> &
  get_external_to_internal_dof_numbering()
  {
    return cluster_tree->get_external_to_internal_dof_numbering();
  }

  const std::vector<types::global_dof_index> &
  get_external_to_internal_dof_numbering() const
  {
    return cluster_tree->get_external_to_internal_dof_numbering();
  }

  DoFToCellTopology<dim, spacedim> &
  get_dof_to_cell_topo()
  {
    return dof_to_cell_topo;
  }

  const DoFToCellTopology<dim, spacedim> &
  get_dof_to_cell_topo() const
  {
    return dof_to_cell_topo;
  }

  std::vector<types::global_dof_index> &
  get_local_to_full_dof_id_map()
  {
    return local_to_full_dof_id_map;
  }

  const std::vector<types::global_dof_index> &
  get_local_to_full_dof_id_map() const
  {
    return local_to_full_dof_id_map;
  }

  types::global_dof_index
  get_n_dofs() const
  {
    return n_dofs;
  }

  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> &
  get_cell_iterators()
  {
    return cell_iterators;
  }

  const std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> &
  get_cell_iterators() const
  {
    return cell_iterators;
  }

private:
  void
  generate_dof_selectors();

  /**
   * Collect cell iterators which are associated with the selected DoFs.
   */
  void
  collect_cell_iterators();

  /**
   * Generate a map from full to local DoF indices and another map from local to
   * global DoF indices.
   *
   * The full DoF indices are the natural indices (starting from zero) for all
   * DoFs in the DoF handler. The local DoF indices are for selected DoFs.
   */
  void
  generate_maps_between_full_and_local_dof_ids();

  void
  build_dof_to_cell_topology();

  const DoFHandler<dim, spacedim> &dof_handler;

  /**
   * Whether the function space is constructed on the whole domain.
   */
  bool is_full_domain;
  /**
   * Whether DoFs at the interface with other material subdomains are selected.
   */
  bool include_boundary_dofs;
  /**
   * Whether limit the support of DoFs at the interface with other material
   * subdomains within the current subdomain. This flag influences the
   * construction of DoF-to-cell topology.
   */
  bool limit_support_in_subdomain;

  /**
   * The set of material ids or a map from material ids to function pointers for
   * the spatial domain on which the function is constructed.
   */
  SearchableMaterialIdContainer material_ids;
  /**
   * A vector of flags indicating selected DoFs for the function space. It is
   * only used when @p is_full_domain is false. The size of this vector is the
   * total number of DoFs in the DoF handler.
   */
  std::vector<bool> dof_selectors;
  /**
   * Number of selected DoFs.
   */
  types::global_dof_index n_dofs;
  /**
   * List of cell iterators which are associated the selected DoFs.
   */
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators;
  std::vector<types::global_dof_index>           full_to_local_dof_id_map;
  std::vector<types::global_dof_index>           local_to_full_dof_id_map;
  DoFToCellTopology<dim, spacedim>               dof_to_cell_topo;
  std::unique_ptr<ClusterTree<spacedim, Number>> cluster_tree;
  std::unique_ptr<ClusterTreeBuilder<spacedim, Number>> cluster_tree_builder;
};


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
template <typename VectorType>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  convert_vector_i2e(const VectorType &vec_i, VectorType &vec_e) const
{
  BEMTools::permute_vector(vec_i,
                           get_external_to_internal_dof_numbering(),
                           vec_e);
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
template <typename VectorType>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  convert_vector_e2i(const VectorType &vec_e, VectorType &vec_i) const
{
  BEMTools::permute_vector(vec_e,
                           get_internal_to_external_dof_numbering(),
                           vec_i);
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
template <typename VectorType>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  extract_selected_dofs(const VectorType &vec, VectorType &subvec) const
{
  Assert(!is_full_domain, ExcInternalError());
  Assert(vec.size() > subvec.size(), ExcInternalError());

  for (types::global_dof_index i = 0; i < subvec.size(); i++)
    subvec(i) = vec(local_to_full_dof_id_map[i]);
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
template <typename VectorType>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  assemble_selected_dofs_into_full_dofs(const VectorType &subvec,
                                        VectorType       &vec) const
{
  AssertDimension(subvec.size(), local_to_full_dof_id_map.size());

  for (types::global_dof_index i = 0; i < subvec.size(); i++)
    vec(local_to_full_dof_id_map[i]) = subvec(i);
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  BEMFunctionSpace(const DoFHandler<dim, spacedim> &dof_handler_,
                   const unsigned int               n_min,
                   const unsigned int               cutoff_level)
  : dof_handler(dof_handler_)
  , is_full_domain(true)
  , include_boundary_dofs(true)
  , limit_support_in_subdomain(false)
  , n_dofs(dof_handler.n_dofs())
{
  cluster_tree_builder =
    std::make_unique<ClusterTreeBuilder<spacedim, Number>>(dof_handler, n_min);
  cluster_tree = cluster_tree_builder->build(cutoff_level);

  build_dof_to_cell_topology();
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  BEMFunctionSpace(const DoFHandler<dim, spacedim>     &dof_handler_,
                   const SearchableMaterialIdContainer &material_ids_,
                   const unsigned int                   n_min,
                   const unsigned int                   cutoff_level,
                   const bool                           include_boundary_dofs_,
                   const bool limit_support_in_subdomain_)
  : dof_handler(dof_handler_)
  , is_full_domain(false)
  , include_boundary_dofs(include_boundary_dofs_)
  , limit_support_in_subdomain(limit_support_in_subdomain_)
  , material_ids(material_ids_)
{
  generate_dof_selectors();
  generate_maps_between_full_and_local_dof_ids();

  cluster_tree_builder = std::make_unique<ClusterTreeBuilder<spacedim, Number>>(
    dof_handler, local_to_full_dof_id_map, n_min);
  cluster_tree = cluster_tree_builder->build(cutoff_level);

  build_dof_to_cell_topology();
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  generate_dof_selectors()
{
  dof_selectors.resize(dof_handler.n_dofs());

  if (include_boundary_dofs)
    n_dofs = DoFToolsExt::extract_material_subdomain_dofs(dof_handler,
                                                          material_ids,
                                                          dof_selectors);
  else
    n_dofs = DoFToolsExt::extract_material_subdomain_dofs_without_boundary_dofs(
      dof_handler, material_ids, dof_selectors);
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  generate_maps_between_full_and_local_dof_ids()
{
  // Vector length initialized to the number of all DoFs in the DoF handler.
  full_to_local_dof_id_map.resize(dof_handler.n_dofs());
  // Vector length initialized to the selected number of DoFs.
  local_to_full_dof_id_map.resize(n_dofs);

  types::global_dof_index local_i = 0;
  for (types::global_dof_index i = 0; i < dof_selectors.size(); i++)
    {
      if (dof_selectors[i])
        {
          local_to_full_dof_id_map[local_i] = i;
          full_to_local_dof_id_map[i]       = local_i;
          local_i++;
        }
    }
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  collect_cell_iterators()
{
  cell_iterators.reserve(dof_handler.get_triangulation().n_active_cells());

  if (is_full_domain || !limit_support_in_subdomain)
    {
      // When @p limit_support_in_subdomain is false, even though the function
      // space is defined on a subdomain, we still need to collect iterators for
      // all cells in the triangulation, because there is one layer of cells
      // extending from the current subdomain into adjacent subdomains, which
      // contain those DoFs at the subdomain interface. At the moment, it is not
      // obvious to directly extract this layer of cells, so we collect
      // iterators for all cells in the triangulation, which will be used for
      // building the DoF-to-cell topology.
      for (const auto &cell : dof_handler.active_cell_iterators())
        cell_iterators.push_back(cell);
    }
  else
    {
      // When the support of DoFs are limited within the subdomain, only the
      // cells with material ids belonging to the subdomain are collected.
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          auto found_iter = material_ids.find(cell->material_id());

          if (found_iter != material_ids.end())
            cell_iterators.push_back(cell);
        }

      cell_iterators.shrink_to_fit();
    }
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          typename Number>
void
BEMFunctionSpace<dim, spacedim, SearchableMaterialIdContainer, Number>::
  build_dof_to_cell_topology()
{
  collect_cell_iterators();

  if (is_full_domain)
    DoFToolsExt::build_dof_to_cell_topology(dof_to_cell_topo,
                                            cell_iterators,
                                            dof_handler);
  else
    DoFToolsExt::build_dof_to_cell_topology(dof_to_cell_topo,
                                            cell_iterators,
                                            dof_handler,
                                            dof_selectors);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_BEM_FUNCTION_SPACE_H_
