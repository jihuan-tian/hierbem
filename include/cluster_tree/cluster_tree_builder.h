// Copyright (C) 2025-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file cluster_tree_builder.h
 * @brief Definition of a class for building a cluster tree.
 *
 * @date 2025-11-14
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_CLUSTER_TREE_CLUSTER_TREE_BUILDER_H_
#define HIERBEM_INCLUDE_CLUSTER_TREE_CLUSTER_TREE_BUILDER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q.h>

#include <memory>
#include <vector>

#include "cluster_tree/cluster_tree.h"
#include "config.h"
#include "dofs/dof_tools_ext.h"
#include "utilities/unary_template_arg_containers.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * @brief Class for building a cluster tree.
 *
 * Here we only use the first order Lagrange mapping, i.e. <tt>MappingQ(1)</tt>,
 * for extracting coordinates of DoF support points, which is good enough for
 * cluster partition.
 */
template <int spacedim, typename Number = double>
class ClusterTreeBuilder
{
public:
  /**
   * @brief Create a builder for constructing a cluster tree on the whole
   * domain.
   * @tparam dim
   * @param dof_handler
   * @param n_min_
   */
  template <int dim>
  ClusterTreeBuilder(const DoFHandler<dim, spacedim> &dof_handler,
                     const unsigned int               n_min_);

  /**
   * @brief Create a builder for constructing a cluster tree on a material
   * subdomain.
   * @tparam dim
   * @param dof_handler
   * @param local_to_full_dof_id_map
   * @param n_min_
   */
  template <int dim>
  ClusterTreeBuilder(
    const DoFHandler<dim, spacedim>            &dof_handler,
    const std::vector<types::global_dof_index> &local_to_full_dof_id_map,
    const unsigned int                          n_min_);

  /**
   * Build a cluster tree and return it as a unique smart pointer, since it will
   * be associated with a unique function space. This function runs with TBB
   * parallelization.
   *
   * @param cutoff_level When the level of a cluster is smaller then this level,
   * the partition from this cluster will be sent to a TBB task.
   */
  std::unique_ptr<ClusterTree<spacedim, Number>>
  build(const unsigned int cutoff_level = 0);

  std::vector<Point<spacedim, Number>> &
  get_support_points()
  {
    return support_points;
  }

  const std::vector<Point<spacedim, Number>> &
  get_support_points() const
  {
    return support_points;
  }

  std::vector<Number> &
  get_dof_support_set_diameters()
  {
    return dof_support_set_diameters;
  }

  const std::vector<Number> &
  get_dof_support_set_diameters() const
  {
    return dof_support_set_diameters;
  }

private:
  /**
   * List of support points associated with this cluster tree.
   */
  std::vector<Point<spacedim, Number>> support_points;
  /**
   * List of DoF indices starting from zero.
   *
   * When the function space to which this cluster tree belongs is defined on a
   * subdomain, these DoF indices are local indices, not full indices.
   */
  std::vector<types::global_dof_index> dof_indices;
  /**
   * List of support set diameters for basis functions at support points.
   */
  std::vector<Number> dof_support_set_diameters;
  /**
   * Minimum number of DoFs in a cluster.
   *
   * When the actual number of DoFs in a cluster is smaller than <tt>n_min</tt>,
   * the cluster is a leaf node in the cluster tree.
   */
  unsigned int n_min;
};


template <int spacedim, typename Number>
template <int dim>
ClusterTreeBuilder<spacedim, Number>::ClusterTreeBuilder(
  const DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int               n_min_)
  : n_min(n_min_)
{
  Assert(dof_handler.get_fe().has_support_points(), ExcInternalError());

  const types::global_dof_index n_dofs = dof_handler.n_dofs();

  // Get the coordinates for all support points.
  support_points.resize(n_dofs);
  DoFToolsExt::map_dofs_to_support_points(MappingQ<dim, spacedim>(1),
                                          dof_handler,
                                          support_points);

  // Generate a list of DoF indices starting from zero.
  dof_indices.resize(n_dofs);
  gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

  // Calculate the DoF support set diameter at each support point.
  dof_support_set_diameters.assign(n_dofs, 0);
  DoFToolsExt::map_dofs_to_support_set_diameters(dof_handler,
                                                 dof_support_set_diameters);
}


template <int spacedim, typename Number>
template <int dim>
ClusterTreeBuilder<spacedim, Number>::ClusterTreeBuilder(
  const DoFHandler<dim, spacedim>            &dof_handler,
  const std::vector<types::global_dof_index> &local_to_full_dof_id_map,
  const unsigned int                          n_min_)
  : n_min(n_min_)
{
  Assert(dof_handler.get_fe().has_support_points(), ExcInternalError());

  const types::global_dof_index n_dofs = local_to_full_dof_id_map.size();

  // Get the coordinates for all support points in the material subdomain.
  support_points.resize(n_dofs);
  DoFToolsExt::map_dofs_to_support_points(MappingQ<dim, spacedim>(1),
                                          dof_handler,
                                          local_to_full_dof_id_map,
                                          support_points);

  // Generate a list of DoF indices starting from zero.
  dof_indices.resize(n_dofs);
  gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

  // Calculate the average mesh cell size at each support point.
  dof_support_set_diameters.assign(n_dofs, 0);
  DoFToolsExt::map_dofs_to_support_set_diameters(dof_handler,
                                                 local_to_full_dof_id_map,
                                                 dof_support_set_diameters);
}


template <int spacedim, typename Number>
std::unique_ptr<ClusterTree<spacedim, Number>>
ClusterTreeBuilder<spacedim, Number>::build(const unsigned int cutoff_level)
{
  // Create a cluster tree for all the DoF indices.
  auto cluster_tree = std::make_unique<ClusterTree<spacedim, Number>>(
    dof_indices, support_points, dof_support_set_diameters, n_min);
  // Partition the cluster tree.
  cluster_tree->partition(support_points,
                          dof_support_set_diameters,
                          cutoff_level);

  // Clear intermediate data.
  support_points.clear();
  dof_indices.clear();
  dof_support_set_diameters.clear();

  return cluster_tree;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CLUSTER_TREE_CLUSTER_TREE_BUILDER_H_
