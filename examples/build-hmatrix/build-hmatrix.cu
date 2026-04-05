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
 * @file build-hmatrix.cu
 * @brief Example for building an H-matrix.
 *
 * @ingroup examples
 * @author Jihuan Tian
 * @date 2025-10-23
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "bem/bem_tools.h"
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
#include "utilities/number_traits.h"
#include "utilities/unary_template_arg_containers.h"

using namespace dealii;
using namespace HierBEM;
using namespace HierBEM::PlatformShared::LaplaceKernel;

// Builder class for cluster tree.
template <int spacedim>
class ClusterTreeBuilder
{
public:
  template <int dim>
  ClusterTreeBuilder(const Mapping<dim, spacedim>    &mapping,
                     const DoFHandler<dim, spacedim> &dof_handler,
                     const unsigned int               _n_min);

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
  const Mapping<dim, spacedim>    &mapping,
  const DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int               _n_min)
  : n_min(_n_min)
{
  Assert(dof_handler.get_fe().has_support_points(), ExcInternalError());

  const types::global_dof_index n_dofs = dof_handler.n_dofs();

  // Get the coordinates for all support points.
  support_points.resize(n_dofs);
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  // Generate a list of DoF indices starting from zero.
  dof_indices.resize(n_dofs);
  for (types::global_dof_index d = 0; d < n_dofs; d++)
    dof_indices[d] = d;

  // Calculate the average mesh cell size at each support point.
  dof_average_cell_size.assign(n_dofs, 0);
  DoFToolsExt::map_dofs_to_average_cell_size(dof_handler,
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
  BEMFunctionSpace(const DoFHandler<dim, spacedim> &dof_handler_,
                   const unsigned int               n_min);

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

private:
  void
  build_dof_to_cell_topology();

  const DoFHandler<dim, spacedim>              &dof_handler;
  std::unique_ptr<ClusterTree<spacedim>>        cluster_tree;
  std::unique_ptr<ClusterTreeBuilder<spacedim>> cluster_tree_builder;
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators;
  DoFToCellTopology<dim, spacedim> dof_to_cell_topo;
};


template <int dim, int spacedim>
BEMFunctionSpace<dim, spacedim>::BEMFunctionSpace(
  const DoFHandler<dim, spacedim> &dof_handler_,
  const unsigned int               n_min)
  : dof_handler(dof_handler_)
{
  cluster_tree_builder =
    std::make_unique<ClusterTreeBuilder<spacedim>>(MappingQ<dim, spacedim>(1),
                                                   dof_handler,
                                                   n_min);
  cluster_tree = cluster_tree_builder->build();

  build_dof_to_cell_topology();
}


template <int dim, int spacedim>
void
BEMFunctionSpace<dim, spacedim>::build_dof_to_cell_topology()
{
  cell_iterators.reserve(dof_handler.get_triangulation().n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    cell_iterators.push_back(cell);
  DoFToolsExt::build_dof_to_cell_topology(dof_to_cell_topo,
                                          cell_iterators,
                                          dof_handler);
}


// Class for a bilinear form, which involves a trial space and a test space.
//
// As a convention, the trial space is placed before the test space when we
// define a bilinear form.
template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType  = double,
          typename KernelNumberType = double>
class BEMBilinearForm
{
public:
  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;

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
  build_hmatrix(
    const ConfHMatrix                       &hmat_params,
    const ConfSauterQuadNearField           &sauter_quad_near_field_params,
    const ConfSauterQuadFarField            &sauter_quad_far_field_params,
    const ConfParallelization               &parallel_params,
    const DeviceNumberType<KernelNumberType> kernel_factor,
    const SauterQuadratureRule<dim>         &sauter_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                               &material_id_to_mapping_index,
    const Table<2, Point<spacedim, real_type>> &mapping_support_point_table,
    SubdomainTopology<dim, spacedim>           &subdomain_topology);

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
    const Table<2, Point<spacedim, real_type>> &mapping_support_point_table,
    SubdomainTopology<dim, spacedim>           &subdomain_topology);

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
  // When the kernel function is symmetric and the trial space and test space is
  // a same space, the bilinear form is symmetric. Hence, the discretized
  // H-matrix should also be symmetric.
  bool                                        is_symmetric;
  std::unique_ptr<BlockClusterTree<spacedim>> block_cluster_tree;
};


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
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
          template <int, typename>
          typename KernelFunctionType,
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
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
BEMBilinearForm<dim,
                spacedim,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  build_hmatrix(
    const ConfHMatrix                       &hmat_params,
    const ConfSauterQuadNearField           &sauter_quad_near_field_params,
    const ConfSauterQuadFarField            &sauter_quad_far_field_params,
    const ConfParallelization               &parallel_params,
    const DeviceNumberType<KernelNumberType> kernel_factor,
    const SauterQuadratureRule<dim>         &sauter_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                               &material_id_to_mapping_index,
    const Table<2, Point<spacedim, real_type>> &mapping_support_point_table,
    SubdomainTopology<dim, spacedim>           &subdomain_topology)
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
    nullptr,
    nullptr,
    test_space.get_internal_to_external_dof_numbering(),
    trial_space.get_internal_to_external_dof_numbering(),
    mappings,
    material_id_to_mapping_index,
    mapping_support_point_table,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    is_symmetric);

  return hmat;
}


template <int dim,
          int spacedim,
          template <int, typename>
          typename KernelFunctionType,
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
    const Table<2, Point<spacedim, real_type>> &mapping_support_point_table,
    SubdomainTopology<dim, spacedim>           &subdomain_topology)
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
    nullptr,
    nullptr,
    test_space.get_internal_to_external_dof_numbering(),
    trial_space.get_internal_to_external_dof_numbering(),
    mappings,
    material_id_to_mapping_index,
    mapping_support_point_table,
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    is_symmetric);

  return hmat;
}

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Read the triangulation.
  Triangulation<dim, spacedim> tria;
  std::ifstream mesh_in(HBEM_TEST_MODEL_DIR "two-spheres-fine.msh");
  read_msh(mesh_in, tria);
  // Generate surface-to-volume and volume-to-surface topology.
  SubdomainTopology<dim, spacedim> subdomain_topology;
  subdomain_topology.generate_topology(HBEM_TEST_MODEL_DIR "two-spheres.brep",
                                       HBEM_TEST_MODEL_DIR "two-spheres.msh");

  // Define manifolds for the two spheres.
  const double                                            inter_distance = 8.0;
  std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
  Manifold<dim, spacedim>                                *left_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(-inter_distance / 2.0, 0, 0));
  Manifold<dim, spacedim> *right_sphere_manifold =
    new SphericalManifold<dim, spacedim>(
      Point<spacedim>(inter_distance / 2.0, 0, 0));
  manifolds[0] = left_sphere_manifold;
  manifolds[1] = right_sphere_manifold;

  // Assign manifold ids to surface entities in the CAD model.
  std::map<EntityTag, types::manifold_id> manifold_description;
  manifold_description[1] = 0;
  manifold_description[2] = 1;

  // Assign manifolds to the triangulation.
  for (auto &cell : tria.active_cell_iterators())
    cell->set_all_manifold_ids(manifold_description[cell->material_id()]);

  for (const auto &m : manifolds)
    tria.set_manifold(m.first, *m.second);

  // Define mappings up to the second order for describing the curved surface.
  std::vector<MappingInfo<dim, spacedim> *> mappings(2);
  for (unsigned int i = 1; i <= 2; i++)
    mappings[i - 1] = new MappingInfo<dim, spacedim>(i);

  // Construct a map from material ids to mapping indices.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  material_id_to_mapping_index[1] = 1;
  material_id_to_mapping_index[2] = 1;

  Table<2, Point<spacedim>> tria_mapping_support_points;
  BEMTools::compute_mapping_support_points_for_tria(
    tria, mappings, material_id_to_mapping_index, tria_mapping_support_points);

  // Parameters for building H-matrices.
  ConfHMatrix             hmat_params{32, 32, 0.8, 5, 0.01, false};
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
  BEMFunctionSpace<dim, spacedim> H_half(
    dof_handler_H_half, static_cast<unsigned int>(hmat_params.n_min_for_ct));

  // Create a discontinuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{-1/2}(\Gamma)\f$ space.
  FE_DGQ<dim, spacedim>     fe_H_minus_half(0);
  DoFHandler<dim, spacedim> dof_handler_H_minus_half(tria);
  dof_handler_H_minus_half.distribute_dofs(fe_H_minus_half);
  BEMFunctionSpace<dim, spacedim> H_minus_half(dof_handler_H_minus_half,
                                               static_cast<unsigned int>(
                                                 hmat_params.n_min_for_ct));

  // Create a bilinear form \f$b_V: H^{-1/2}(\Gamma)\times H^{-1/2}(\Gamma)
  // \rightarrow \mathbb{R}\f$ for the single layer potential operator \f$V\f$.
  BEMBilinearForm<dim, spacedim, SingleLayerKernel> bV(H_minus_half,
                                                       H_minus_half);
  bV.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));
  // Create a bilinear form \f$b_{\frac{1}{2}I+K}: H^{1/2}{\Gamma}\times
  // H^{-1/2}{\Gamma} \rightarrow \mathbb{R}\f$ for the double layer potential
  // operator plus a scaled identity operator \f$\frac{1}{2}I+K\f$. This
  // bilinear form is needed to build the right hand side vector of the Laplace
  // equation with a Dirichlet boundary condition.
  BEMBilinearForm<dim, spacedim, DoubleLayerKernel> bIK(H_half, H_minus_half);
  bIK.build_block_cluster_tree(
    hmat_params.eta, static_cast<unsigned int>(hmat_params.n_min_for_bct));

  // Build an H-matrix for bV.
  std::unique_ptr<HMatrix<spacedim, double>> V =
    bV.build_hmatrix(hmat_params,
                     sauter_quad_near_field_params,
                     sauter_quad_far_field_params,
                     parallel_params,
                     1.0,
                     SauterQuadratureRule<dim>(5, 4, 4, 3),
                     mappings,
                     material_id_to_mapping_index,
                     tria_mapping_support_points,
                     subdomain_topology);
  // Build an H-matrix for bIK.
  std::unique_ptr<HMatrix<spacedim, double>> IK =
    bIK.build_hmatrix_with_mass_matrix(hmat_params,
                                       sauter_quad_near_field_params,
                                       sauter_quad_far_field_params,
                                       parallel_params,
                                       1.0,
                                       0.5,
                                       SauterQuadratureRule<dim>(5, 4, 4, 3),
                                       QGauss<dim>(2),
                                       mappings,
                                       material_id_to_mapping_index,
                                       tria_mapping_support_points,
                                       subdomain_topology);

  // Print out the leaf set information of H-matrices. For each leaf node,
  // the DoF index ranges in the block cluster, near field/far field flag and
  // matrix rank are printed.
  std::ofstream leaf_set("V-leaf-set.dat");
  V->write_leaf_set_by_iteration(leaf_set);
  leaf_set.close();

  leaf_set.open("IK-leaf-set.dat");
  IK->write_leaf_set_by_iteration(leaf_set);
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
