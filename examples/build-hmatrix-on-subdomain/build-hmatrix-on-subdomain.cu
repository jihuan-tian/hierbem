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
 * @file build-hmatrix-on-subdomain.cu
 * @brief Example for building an H-matrix on a subdomain which is specified by
 * a set of material ids.
 *
 * @ingroup examples
 * @author Jihuan Tian
 * @date 2025-11-04
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "cad_mesh/gmsh_manipulation.h"
#include "cad_mesh/subdomain_topology.h"
#include "cluster_tree/block_cluster_tree.h"
#include "cluster_tree/cluster_tree.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "dofs/dof_to_cell_topology.h"
#include "dofs/dof_tools_ext.h"
#include "grid/grid_in_ext.h"
#include "hbem_test_config.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_support.h"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/generic_functors.h"
#include "utilities/number_traits.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;
using namespace HierBEM::PlatformShared::LaplaceKernel;

// Builder class for cluster tree.
//
// Here we only use the first order mapping for extracting coordinates of DoF
// support points, which is good enough.
template <int spacedim>
class ClusterTreeBuilder
{
public:
  // Create a builder for constructing a cluster tree on the whole domain.
  template <int dim>
  ClusterTreeBuilder(const DoFHandler<dim, spacedim> &dof_handler,
                     const unsigned int               n_min_);

  // Create a builder for constructing a cluster tree on a material subdomain.
  template <int dim>
  ClusterTreeBuilder(
    const DoFHandler<dim, spacedim>            &dof_handler,
    const std::vector<types::global_dof_index> &local_to_full_dof_id_map,
    const unsigned int                          n_min_);

  // Build a cluster tree and return it as a unique smart pointer, since it will
  // be associated with a unique function space.
  std::unique_ptr<ClusterTree<spacedim>>
  build() const;

  std::vector<Point<spacedim>> &
  get_support_points()
  {
    return support_points;
  }

  const std::vector<Point<spacedim>> &
  get_support_points() const
  {
    return support_points;
  }

  std::vector<double> &
  get_dof_average_cell_size()
  {
    return dof_average_cell_size;
  }

  const std::vector<double> &
  get_dof_average_cell_size() const
  {
    return dof_average_cell_size;
  }

private:
  // List of support points in the finite element associated with this cluster
  // tree.
  std::vector<Point<spacedim>> support_points;
  // List of DoF indices in the finite element, starting from zero.
  std::vector<types::global_dof_index> dof_indices;
  // List of average cell size estimated at support points.
  std::vector<double> dof_average_cell_size;
  // Minimum number of DoFs in a cluster. When the actual number of DoFs in a
  // cluster is smaller than @p n_min , the cluster is a leaf node in the
  // cluster tree.
  unsigned int n_min;
};


template <int spacedim>
template <int dim>
ClusterTreeBuilder<spacedim>::ClusterTreeBuilder(
  const DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int               n_min_)
  : n_min(n_min_)
{
  Assert(dof_handler.get_fe().has_support_points(), ExcInternalError());

  const types::global_dof_index n_dofs = dof_handler.n_dofs();

  // Get the coordinates for all support points.
  support_points.resize(n_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ<dim, spacedim>(1),
                                       dof_handler,
                                       support_points);

  // Generate a list of DoF indices starting from zero.
  dof_indices.resize(n_dofs);
  gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

  // Calculate the average mesh cell size at each support point.
  dof_average_cell_size.assign(n_dofs, 0);
  DoFToolsExt::map_dofs_to_average_cell_size(dof_handler,
                                             dof_average_cell_size);
}


template <int spacedim>
template <int dim>
ClusterTreeBuilder<spacedim>::ClusterTreeBuilder(
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
  dof_average_cell_size.assign(n_dofs, 0);
  DoFToolsExt::map_dofs_to_average_cell_size(dof_handler,
                                             local_to_full_dof_id_map,
                                             dof_average_cell_size);
}


template <int spacedim>
std::unique_ptr<ClusterTree<spacedim>>
ClusterTreeBuilder<spacedim>::build() const
{
  // Create a cluster tree for all the DoF indices.
  auto cluster_tree = std::make_unique<ClusterTree<spacedim>>(
    dof_indices, support_points, dof_average_cell_size, n_min);
  // Partition the cluster tree.
  cluster_tree->partition(support_points, dof_average_cell_size);

  return cluster_tree;
}


// Class for a function space used in BEM, which contains a cluster tree.
template <int dim, int spacedim>
class BEMFunctionSpace
{
public:
  // Construct a function space on the whole domain.
  BEMFunctionSpace(const DoFHandler<dim, spacedim> &dof_handler_,
                   const unsigned int               n_min);

  // Construct a function space on a material subdomain.
  BEMFunctionSpace(const DoFHandler<dim, spacedim>    &dof_handler_,
                   const unsigned int                  n_min,
                   const std::set<types::material_id> &material_ids_,
                   const bool include_boundary_dofs_      = true,
                   const bool limit_support_in_subdomain_ = false);

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

  ClusterTree<spacedim> &
  get_cluster_tree()
  {
    return *cluster_tree;
  }

  const ClusterTree<spacedim> &
  get_cluster_tree() const
  {
    return *cluster_tree;
  }

  ClusterTreeBuilder<spacedim> &
  get_cluster_tree_builder()
  {
    return *cluster_tree_builder;
  }

  const ClusterTreeBuilder<spacedim> &
  get_cluster_tree_builder() const
  {
    return *cluster_tree_builder;
  }

  std::vector<Point<spacedim>> &
  get_support_points()
  {
    return cluster_tree_builder->get_support_points();
  }

  const std::vector<Point<spacedim>> &
  get_support_points() const
  {
    return cluster_tree_builder->get_support_points();
  }

  std::vector<double> &
  get_dof_average_cell_size()
  {
    return cluster_tree_builder->get_dof_average_cell_size();
  }

  const std::vector<double> &
  get_dof_average_cell_size() const
  {
    return cluster_tree_builder->get_dof_average_cell_size();
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

private:
  void
  generate_dof_selectors();

  // Collect cell iterators which are associated with the selected DoFs.
  void
  collect_cell_iterators();

  // The full DoF indices are the natural indices (starting from zero) for all
  // DoFs in the DoF handler. The local DoF indices are for selected DoFs.
  void
  generate_maps_between_full_and_local_dof_ids();

  void
  build_dof_to_cell_topology();

  const DoFHandler<dim, spacedim> &dof_handler;

  // Whether the function space is constructed on the whole domain.
  bool is_full_domain;
  // Whether DoFs at the interface with other material subdomains are selected.
  bool include_boundary_dofs;
  // Whether limit the support of DoFs at the interface with other material
  // subdomains within the current subdomain. This flag influences the
  // construction of DoF-to-cell topology.
  bool limit_support_in_subdomain;

  // The set of material ids for the spatial domain on which the function is
  // constructed. When it is an empty set, the function space is on the whole
  // triangulation and the flag @p is_full_domain is true.
  std::set<types::material_id> material_ids;
  // A vector of flags indicating selected DoFs for the function space. It is
  // only used when @p is_full_domain is false. The size of this vector is the
  // total number of DoFs in the DoF handler.
  std::vector<bool> dof_selectors;
  // Number of selected DoFs.
  types::global_dof_index n_dofs;
  // List of cell iterators which are associated the selected DoFs.
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators;
  std::vector<types::global_dof_index>          full_to_local_dof_id_map;
  std::vector<types::global_dof_index>          local_to_full_dof_id_map;
  DoFToCellTopology<dim, spacedim>              dof_to_cell_topo;
  std::unique_ptr<ClusterTree<spacedim>>        cluster_tree;
  std::unique_ptr<ClusterTreeBuilder<spacedim>> cluster_tree_builder;
};


template <int dim, int spacedim>
BEMFunctionSpace<dim, spacedim>::BEMFunctionSpace(
  const DoFHandler<dim, spacedim> &dof_handler_,
  const unsigned int               n_min)
  : dof_handler(dof_handler_)
  , is_full_domain(true)
  , include_boundary_dofs(true)
  , limit_support_in_subdomain(false)
  , n_dofs(dof_handler.n_dofs())
{
  cluster_tree_builder =
    std::make_unique<ClusterTreeBuilder<spacedim>>(dof_handler, n_min);
  cluster_tree = cluster_tree_builder->build();

  build_dof_to_cell_topology();
}


template <int dim, int spacedim>
BEMFunctionSpace<dim, spacedim>::BEMFunctionSpace(
  const DoFHandler<dim, spacedim>    &dof_handler_,
  const unsigned int                  n_min,
  const std::set<types::material_id> &material_ids_,
  const bool                          include_boundary_dofs_,
  const bool                          limit_support_in_subdomain_)
  : dof_handler(dof_handler_)
  , is_full_domain(false)
  , include_boundary_dofs(include_boundary_dofs_)
  , limit_support_in_subdomain(limit_support_in_subdomain_)
  , material_ids(material_ids_)
{
  generate_dof_selectors();
  generate_maps_between_full_and_local_dof_ids();

  cluster_tree_builder =
    std::make_unique<ClusterTreeBuilder<spacedim>>(dof_handler,
                                                   local_to_full_dof_id_map,
                                                   n_min);
  cluster_tree = cluster_tree_builder->build();

  build_dof_to_cell_topology();
}


template <int dim, int spacedim>
void
BEMFunctionSpace<dim, spacedim>::generate_dof_selectors()
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


template <int dim, int spacedim>
void
BEMFunctionSpace<dim, spacedim>::generate_maps_between_full_and_local_dof_ids()
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


template <int dim, int spacedim>
void
BEMFunctionSpace<dim, spacedim>::collect_cell_iterators()
{
  cell_iterators.reserve(dof_handler.get_triangulation().n_active_cells());

  if (is_full_domain || !limit_support_in_subdomain)
    {
      for (const auto &cell : dof_handler.active_cell_iterators())
        cell_iterators.push_back(cell);
    }
  else if (limit_support_in_subdomain)
    {
      // When the support of DoFs are limited within the subdomain, only the
      // cells with material ids belonging to the subdomain are collected.
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          auto found_iter = material_ids.find(cell->material_id());

          if (found_iter != material_ids.end())
            cell_iterators.push_back(cell);
        }
    }
}


template <int dim, int spacedim>
void
BEMFunctionSpace<dim, spacedim>::build_dof_to_cell_topology()
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


// Class for a bilinear form, which involves a trial space and a test space.
//
// As a convention, the trial space is placed before the test space when we
// define a bilinear form.
template <int dim,
          int spacedim,
          template <int, typename> typename KernelFunctionType,
          typename RangeNumberType  = double,
          typename KernelNumberType = double>
class BEMBilinearForm
{
public:
  BEMBilinearForm(const BEMFunctionSpace<dim, spacedim> &trial_space_,
                  const BEMFunctionSpace<dim, spacedim> &test_space_);

  // @param eta Admissibility constant. Englaring this parameter will make more
  // leaf nodes of the block cluster tree be far field.
  // @param n_min Minimum number of DoFs in a cluster. For a block cluster
  // \f$\tau\times \sigma\f$, whenever the cardinality of \f$\tau\f$ or
  // \f$\sigma\f$ is smaller than @p n_min , this block cluster is a near field
  // node.
  void
  build_block_cluster_tree(const double eta, const unsigned int n_min);

  // Build an H-matrix for the bilinear form.
  std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
  build_hmatrix(const ConfHMatrix             &hmat_params,
                const ConfSauterQuadNearField &sauter_quad_near_field_params,
                const ConfSauterQuadFarField  &sauter_quad_far_field_params,
                const ConfParallelization     &parallel_params,
                const DeviceNumberType<KernelNumberType> kernel_factor,
                const SauterQuadratureRule<dim>         &sauter_quad_rule,
                const std::vector<MappingInfo<dim, spacedim> *> &mappings,
                const std::map<types::material_id, unsigned int>
                                                 &material_id_to_mapping_index,
                SubdomainTopology<dim, spacedim> &subdomain_topology);

  // Build an H-matrix for the bilinear form and add a mass matrix directly into
  // it.
  std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
  build_hmatrix_with_mass_matrix(
    const ConfHMatrix                       &hmat_params,
    const ConfSauterQuadNearField           &sauter_quad_near_field_params,
    const ConfSauterQuadFarField            &sauter_quad_far_field_params,
    const ConfParallelization               &parallel_params,
    const DeviceNumberType<KernelNumberType> kernel_factor,
    const typename numbers::NumberTraits<RangeNumberType>::real_type
                                                     mass_matrix_factor,
    const SauterQuadratureRule<dim>                 &sauter_quad_rule,
    const QGauss<dim>                               &mass_matrix_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                     &material_id_to_mapping_index,
    SubdomainTopology<dim, spacedim> &subdomain_topology);

  ClusterTree<spacedim> &
  get_cluster_tree_trial_space()
  {
    return trial_space.get_cluster_tree();
  }

  const ClusterTree<spacedim> &
  get_cluster_tree_trial_space() const
  {
    return trial_space.get_cluster_tree();
  }

  ClusterTree<spacedim> &
  get_cluster_tree_test_space()
  {
    return test_space.get_cluster_tree();
  }

  const ClusterTree<spacedim> &
  get_cluster_tree_test_space() const
  {
    return test_space.get_cluster_tree();
  }

  BlockClusterTree<spacedim> &
  get_block_cluster_tree()
  {
    return *block_cluster_tree;
  }

  const BlockClusterTree<spacedim> &
  get_block_cluster_tree() const
  {
    return *block_cluster_tree;
  }

private:
  KernelFunctionType<spacedim, DeviceNumberType<KernelNumberType>> kernel;
  const BEMFunctionSpace<dim, spacedim>                           &trial_space;
  const BEMFunctionSpace<dim, spacedim>                           &test_space;
  // Whether the bilinear form is symmetric, i.e. the trial space is the same as
  // the test space.
  bool                                        is_symmetric;
  std::unique_ptr<BlockClusterTree<spacedim>> block_cluster_tree;
};


template <int dim,
          int spacedim,
          template <int, typename> typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
BEMBilinearForm<dim,
                spacedim,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  BEMBilinearForm(const BEMFunctionSpace<dim, spacedim> &trial_space_,
                  const BEMFunctionSpace<dim, spacedim> &test_space_)
  : trial_space(trial_space_)
  , test_space(test_space_)
  , is_symmetric(kernel.is_symmetric() && (&trial_space == &test_space))
{}


template <int dim,
          int spacedim,
          template <int, typename> typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
void
BEMBilinearForm<dim,
                spacedim,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::build_block_cluster_tree(const double eta,
                                                            const unsigned int
                                                              n_min)
{
  // When building a block cluster tree, the test space appears before the trial
  // space, since the test space is related to matrix rows, while the trial
  // space is related to matrix columns.
  block_cluster_tree = std::make_unique<BlockClusterTree<spacedim>>(
    test_space.get_cluster_tree(), trial_space.get_cluster_tree(), eta, n_min);

  if (&trial_space == &test_space)
    block_cluster_tree->partition(
      trial_space.get_internal_to_external_dof_numbering(),
      trial_space.get_support_points(),
      trial_space.get_dof_average_cell_size());
  else
    block_cluster_tree->partition(
      test_space.get_internal_to_external_dof_numbering(),
      trial_space.get_internal_to_external_dof_numbering(),
      test_space.get_support_points(),
      trial_space.get_support_points(),
      test_space.get_dof_average_cell_size(),
      trial_space.get_dof_average_cell_size());
}


template <int dim,
          int spacedim,
          template <int, typename> typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
BEMBilinearForm<dim,
                spacedim,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  build_hmatrix(const ConfHMatrix             &hmat_params,
                const ConfSauterQuadNearField &sauter_quad_near_field_params,
                const ConfSauterQuadFarField  &sauter_quad_far_field_params,
                const ConfParallelization     &parallel_params,
                const DeviceNumberType<KernelNumberType> kernel_factor,
                const SauterQuadratureRule<dim>         &sauter_quad_rule,
                const std::vector<MappingInfo<dim, spacedim> *> &mappings,
                const std::map<types::material_id, unsigned int>
                                                 &material_id_to_mapping_index,
                SubdomainTopology<dim, spacedim> &subdomain_topology)
{
  HMatrixSupport::Property  property = is_symmetric ?
                                         HMatrixSupport::Property::symmetric :
                                         HMatrixSupport::Property::general;
  HMatrixSupport::BlockType block_type =
    HMatrixSupport::BlockType::diagonal_block;
  auto hmat = std::make_unique<HMatrix<spacedim, RangeNumberType>>(
    *block_cluster_tree,
    static_cast<unsigned int>(hmat_params.max_rank),
    property,
    block_type);

  fill_hmatrix_with_aca_plus_smp<dim,
                                 spacedim,
                                 KernelFunctionType,
                                 RangeNumberType,
                                 KernelNumberType,
                                 SurfaceNormalDetector<dim, spacedim>>(
    *hmat,
    hmat_params,
    sauter_quad_near_field_params,
    sauter_quad_far_field_params,
    parallel_params,
    kernel,
    kernel_factor,
    test_space.get_dof_to_cell_topo(),
    trial_space.get_dof_to_cell_topo(),
    sauter_quad_rule,
    test_space.get_dof_handler(),
    trial_space.get_dof_handler(),
    test_space.get_is_full_domain() ?
      nullptr :
      &test_space.get_local_to_full_dof_id_map(),
    trial_space.get_is_full_domain() ?
      nullptr :
      &trial_space.get_local_to_full_dof_id_map(),
    test_space.get_internal_to_external_dof_numbering(),
    trial_space.get_internal_to_external_dof_numbering(),
    mappings,
    material_id_to_mapping_index,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    is_symmetric);

  return hmat;
}


template <int dim,
          int spacedim,
          template <int, typename> typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
BEMBilinearForm<dim,
                spacedim,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  build_hmatrix_with_mass_matrix(
    const ConfHMatrix                       &hmat_params,
    const ConfSauterQuadNearField           &sauter_quad_near_field_params,
    const ConfSauterQuadFarField            &sauter_quad_far_field_params,
    const ConfParallelization               &parallel_params,
    const DeviceNumberType<KernelNumberType> kernel_factor,
    const typename numbers::NumberTraits<RangeNumberType>::real_type
                                                     mass_matrix_factor,
    const SauterQuadratureRule<dim>                 &sauter_quad_rule,
    const QGauss<dim>                               &mass_matrix_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                     &material_id_to_mapping_index,
    SubdomainTopology<dim, spacedim> &subdomain_topology)
{
  HMatrixSupport::Property  property = is_symmetric ?
                                         HMatrixSupport::Property::symmetric :
                                         HMatrixSupport::Property::general;
  HMatrixSupport::BlockType block_type =
    HMatrixSupport::BlockType::diagonal_block;
  auto hmat = std::make_unique<HMatrix<spacedim, RangeNumberType>>(
    *block_cluster_tree,
    static_cast<unsigned int>(hmat_params.max_rank),
    property,
    block_type);

  fill_hmatrix_with_aca_plus_smp<dim,
                                 spacedim,
                                 KernelFunctionType,
                                 RangeNumberType,
                                 KernelNumberType,
                                 SurfaceNormalDetector<dim, spacedim>>(
    *hmat,
    hmat_params,
    sauter_quad_near_field_params,
    sauter_quad_far_field_params,
    parallel_params,
    kernel,
    kernel_factor,
    mass_matrix_factor,
    test_space.get_dof_to_cell_topo(),
    trial_space.get_dof_to_cell_topo(),
    sauter_quad_rule,
    mass_matrix_quad_rule,
    test_space.get_dof_handler(),
    trial_space.get_dof_handler(),
    test_space.get_is_full_domain() ?
      nullptr :
      &test_space.get_local_to_full_dof_id_map(),
    trial_space.get_is_full_domain() ?
      nullptr :
      &trial_space.get_local_to_full_dof_id_map(),
    test_space.get_internal_to_external_dof_numbering(),
    trial_space.get_internal_to_external_dof_numbering(),
    mappings,
    material_id_to_mapping_index,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    is_symmetric);

  return hmat;
}


template <int dim, int spacedim>
void
visualize_dofs_in_function_space(const std::string &file_basename,
                                 const BEMFunctionSpace<dim, spacedim> &space)
{
  const types::global_dof_index n_dofs = space.get_dof_handler().n_dofs();
  const std::vector<bool>      &dof_selectors = space.get_dof_selectors();
  Vector<double>                dof_markers(n_dofs);
  for (types::global_dof_index i = 0; i < n_dofs; i++)
    if (dof_selectors[i])
      dof_markers(i) = 1.0;
    else
      dof_markers(i) = 0;

  std::ofstream          vtk_output(file_basename + ".vtk");
  DataOut<dim, spacedim> data_out;
  data_out.add_data_vector(space.get_dof_handler(), dof_markers, "dof_support");
  data_out.build_patches();
  data_out.write_vtk(vtk_output);

  std::ofstream                       point_output(file_basename + ".txt");
  const std::vector<Point<spacedim>> &support_points =
    space.get_support_points();
  for (types::global_dof_index i = 0; i < support_points.size(); i++)
    point_output << support_points[i] << "\n";

  point_output.close();
}

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Read the triangulation.
  Triangulation<dim, spacedim> tria;
  std::ifstream                mesh_in(HBEM_TEST_MODEL_DIR "bar.msh");
  read_msh(mesh_in, tria);
  // Generate surface-to-volume and volume-to-surface topology.
  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_topology(HBEM_TEST_MODEL_DIR "bar.brep",
                                       HBEM_TEST_MODEL_DIR "bar.msh");

  // Define manifold for the bar.
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  Manifold<dim, spacedim> *flat_manifold = new FlatManifold<dim, spacedim>();
  manifolds[0]                           = flat_manifold;

  // Assign manifold ids to surface entities in the CAD model.
  std::map<EntityTag, types::manifold_id> manifold_description;
  for (types::material_id i = 1; i <= 6; i++)
    manifold_description[i] = 0;

  // Assign manifolds to the triangulation.
  for (auto &cell : tria.active_cell_iterators())
    cell->set_all_manifold_ids(manifold_description[cell->material_id()]);

  for (const auto &m : manifolds)
    tria.set_manifold(m.first, *m.second);

  // Define only 1st order mapping for flat surfaces
  std::vector<MappingInfo<dim, spacedim> *> mappings(1);
  mappings[0] = new MappingInfo<dim, spacedim>(1);

  // Construct a map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  for (types::material_id i = 1; i <= 6; i++)
    material_id_to_mapping_index[i] = 0;

  // Parameters for building H-matrices.
  ConfHMatrix             hmat_params{4, 4, 1.2, 5, 0.01};
  ConfSauterQuadNearField sauter_quad_near_field_params;
  ConfSauterQuadFarField  sauter_quad_far_field_params;
  ConfParallelization     parallel_params;

  // Set TBB thread num.
  if (parallel_params.tbb_thread_num == -1)
    MultithreadInfo::set_thread_limit(MultithreadInfo::n_threads());
  else
    MultithreadInfo::set_thread_limit(parallel_params.tbb_thread_num);

  // Initialize CUDA stack size and device properties.
  initCudaRuntime(parallel_params);

  // Create a continuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{1/2}(\Gamma)\f$.
  FE_Q<dim, spacedim>       fe_H_half(1);
  DoFHandler<dim, spacedim> dof_handler_H_half(tria);
  dof_handler_H_half.distribute_dofs(fe_H_half);
  // Define a function space
  // \f$\tilde{H}_h^{1/2}(\Gamma_{\mathrm{D}}^{\ast})\f$.
  BEMFunctionSpace<dim, spacedim> H_half_Gamma_D(dof_handler_H_half,
                                                 static_cast<unsigned int>(
                                                   hmat_params.n_min_for_ct),
                                                 {5, 6},
                                                 true,
                                                 false);
  // Define a function space \f$\tilde{H}_h^{1/2}(\Gamma_{\mathrm{N}})\f$.
  BEMFunctionSpace<dim, spacedim> H_half_Gamma_N(dof_handler_H_half,
                                                 static_cast<unsigned int>(
                                                   hmat_params.n_min_for_ct),
                                                 {1, 2, 3, 4},
                                                 false,
                                                 false);

  // Create a discontinuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{-1/2}(\Gamma)\f$ space.
  FE_DGQ<dim, spacedim>     fe_H_minus_half(0);
  DoFHandler<dim, spacedim> dof_handler_H_minus_half(tria);
  dof_handler_H_minus_half.distribute_dofs(fe_H_minus_half);
  // Define a function space \f$\tilde{H}_h^{-1/2}(\Gamma_{\mathrm{D}})\f$.
  BEMFunctionSpace<dim, spacedim> H_minus_half_Gamma_D(
    dof_handler_H_minus_half,
    static_cast<unsigned int>(hmat_params.n_min_for_ct),
    {5, 6},
    true,
    false);
  // Define a function space \f$\tilde{H}_h^{-1/2}(\Gamma_{\mathrm{N}})\f$.
  BEMFunctionSpace<dim, spacedim> H_minus_half_Gamma_N(
    dof_handler_H_minus_half,
    static_cast<unsigned int>(hmat_params.n_min_for_ct),
    {1, 2, 3, 4},
    true,
    false);

  // Create a bilinear form \f$b_V: b_{V_1}:
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$
  BEMBilinearForm<dim, spacedim, SingleLayerKernel> bV1(H_minus_half_Gamma_D,
                                                        H_minus_half_Gamma_D);
  bV1.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));
  // Create a bilinear form \f$b_{K_1}:
  // \tilde{H}^{1/2}(\Gamma_{\mathrm{N}}) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$.
  BEMBilinearForm<dim, spacedim, DoubleLayerKernel> bK1(H_half_Gamma_N,
                                                        H_minus_half_Gamma_D);
  bK1.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));
  // Create a bilinear form \f$b_{V_2}: H^{-1/2}(\Gamma) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$.
  BEMBilinearForm<dim, spacedim, SingleLayerKernel> bV2(H_minus_half_Gamma_N,
                                                        H_minus_half_Gamma_D);
  bV2.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));
  // Create a bilinear form \f$b_{sigma I_1+K_2}: H^{1/2}(\Gamma) \times
  // \tilde{H}^{-1/2}(\Gamma_{\mathrm{D}}) \rightarrow \mathbb{R}\f$.
  BEMBilinearForm<dim, spacedim, DoubleLayerKernel> bI1K2(H_half_Gamma_D,
                                                          H_minus_half_Gamma_D);
  bI1K2.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));

  // Build an H-matrix for bV1.
  std::unique_ptr<HMatrix<spacedim, double>> V1 =
    bV1.build_hmatrix(hmat_params,
                      sauter_quad_near_field_params,
                      sauter_quad_far_field_params,
                      parallel_params,
                      1.0,
                      SauterQuadratureRule<dim>(5, 4, 4, 3),
                      mappings,
                      material_id_to_mapping_index,
                      subdomain_topology);
  // Build an H-matrix for bK1.
  std::unique_ptr<HMatrix<spacedim, double>> K1 =
    bK1.build_hmatrix(hmat_params,
                      sauter_quad_near_field_params,
                      sauter_quad_far_field_params,
                      parallel_params,
                      1.0,
                      SauterQuadratureRule<dim>(5, 4, 4, 3),
                      mappings,
                      material_id_to_mapping_index,
                      subdomain_topology);
  // Build an H-matrix for bV2.
  std::unique_ptr<HMatrix<spacedim, double>> V2 =
    bV2.build_hmatrix(hmat_params,
                      sauter_quad_near_field_params,
                      sauter_quad_far_field_params,
                      parallel_params,
                      1.0,
                      SauterQuadratureRule<dim>(5, 4, 4, 3),
                      mappings,
                      material_id_to_mapping_index,
                      subdomain_topology);
  // Build an H-matrix for bI1K2.
  std::unique_ptr<HMatrix<spacedim, double>> I1K2 =
    bI1K2.build_hmatrix_with_mass_matrix(hmat_params,
                                         sauter_quad_near_field_params,
                                         sauter_quad_far_field_params,
                                         parallel_params,
                                         1.0,
                                         0.5,
                                         SauterQuadratureRule<dim>(5, 4, 4, 3),
                                         QGauss<dim>(2),
                                         mappings,
                                         material_id_to_mapping_index,
                                         subdomain_topology);

  // Generate visualizations of all function spaces.
  visualize_dofs_in_function_space("H_half_Gamma_D", H_half_Gamma_D);
  visualize_dofs_in_function_space("H_half_Gamma_N", H_half_Gamma_N);
  visualize_dofs_in_function_space("H_minus_half_Gamma_D",
                                   H_minus_half_Gamma_D);
  visualize_dofs_in_function_space("H_minus_half_Gamma_N",
                                   H_minus_half_Gamma_N);

  // Print out the leaf set information of H-matrices. For each leaf node,
  // the DoF index ranges in the block cluster, near field/far field flag and
  // matrix rank are printed.
  std::ofstream leaf_set("V1-leaf-set.dat");
  V1->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("K1-leaf-set.dat");
  K1->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("V2-leaf-set.dat");
  V2->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("I1K2-leaf-set.dat");
  I1K2->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  // Delete manifolds and mappings.
  for (auto &m : manifolds)
    if (m.second != nullptr)
      delete m.second;

  for (auto &m : mappings)
    if (m != nullptr)
      delete m;

  return 0;
}
