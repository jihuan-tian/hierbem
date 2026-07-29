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
 * @file bem_bilinear_form.h
 * @brief Definition of a class for BEM bilinear form.
 *
 * @date 2025-11-14
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_BEM_BEM_BILINEAR_FORM_H_
#define HIERBEM_INCLUDE_BEM_BEM_BILINEAR_FORM_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <map>
#include <memory>
#include <vector>

#include "bem_function_space.h"
#include "bem_tools.h"
#include "cad_mesh/subdomain_topology.h"
#include "cluster_tree/block_cluster_tree.h"
#include "config.h"
#include "config_file/config_structs.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_support.h"
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/number_traits.h"
#include "utilities/unary_template_arg_containers.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Class for a bilinear form, which involves a trial space and a test space.
 *
 * As a convention, the trial space is placed before the test space when we
 * define a bilinear form.
 */
template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType  = double,
          typename KernelNumberType = double>
class BEMBilinearForm
{
public:
  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;

  BEMBilinearForm(const BEMFunctionSpace<dim,
                                         spacedim,
                                         SearchableMaterialIdContainer,
                                         real_type> &trial_space_,
                  const BEMFunctionSpace<dim,
                                         spacedim,
                                         SearchableMaterialIdContainer,
                                         real_type> &test_space_);

  /**
   * Build a block cluster tree.
   * @param eta Admissibility constant. Englaring this parameter will make more
   * leaf nodes of the block cluster tree be far field.
   * @param n_min Minimum number of DoFs in a cluster. For a block cluster
   * \f$\tau\times \sigma\f$, whenever the cardinality of \f$\tau\f$ or
   * \f$\sigma\f$ is smaller than @p n_min , this block cluster is a near field
   * node.
   * @param cutoff_level When the level of a node is less than this value, the
   * partition from its four children will be started as a TBB task.
   */
  void
  build_block_cluster_tree(const real_type    eta,
                           const unsigned int n_min,
                           const unsigned int cutoff_level = 0);

  /**
   * Build an H-matrix for the bilinear form.
   */
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
    const Table<2, Point<spacedim, real_type>> &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<2, Point<spacedim, real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    SubdomainTopology<dim, spacedim>               &subdomain_topology);

  /**
   * Build an H-matrix for the bilinear form and add a mass matrix directly into
   * it.
   */
  std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
  build_hmatrix_with_mass_matrix(
    const ConfHMatrix                       &hmat_params,
    const ConfSauterQuadNearField           &sauter_quad_near_field_params,
    const ConfSauterQuadFarField            &sauter_quad_far_field_params,
    const ConfParallelization               &parallel_params,
    const DeviceNumberType<KernelNumberType> kernel_factor,
    const real_type                          mass_matrix_factor,
    const SauterQuadratureRule<dim>         &sauter_quad_rule,
    const QGauss<dim>                       &mass_matrix_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                               &material_id_to_mapping_index,
    const Table<2, Point<spacedim, real_type>> &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<2, Point<spacedim, real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    SubdomainTopology<dim, spacedim>               &subdomain_topology);

  /**
   * @brief Build an H-matrix for a bilinear form which needs regularization,
   * such as the bilinear form for the hyper singular boundary integral
   * operator. With regularization, the trial and test functions in the double
   * integral will be applied surface curl.
   *
   * One or several stabilization vectors should also be directly built into the
   * H-matrix, when the hyper singular boundary integral operator is not
   * elliptic, such as in the Laplace equation. When in the Helmholtz acoustic
   * equation, such stabilization vectors are not needed, since the hyper
   * singular boundary integral operator is coercive, when the wave number
   * squared is not an eigenvalue of the internal Dirichlet or Neumann problem.
   */
  std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
  build_hmatrix_with_regularization(
    const ConfHMatrix                           &hmat_params,
    const ConfSauterQuadNearField               &sauter_quad_near_field_params,
    const ConfSauterQuadFarField                &sauter_quad_far_field_params,
    const ConfParallelization                   &parallel_params,
    const DeviceNumberType<KernelNumberType>     kernel_factor,
    const std::vector<Vector<KernelNumberType>> &mass_vmult_weq,
    const KernelNumberType                       stabilization_factor,
    const SauterQuadratureRule<dim>             &sauter_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                               &material_id_to_mapping_index,
    const Table<2, Point<spacedim, real_type>> &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<2, Point<spacedim, real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    SubdomainTopology<dim, spacedim>               &subdomain_topology);

  KernelFunctionType<spacedim, DeviceNumberType<KernelNumberType>> &
  get_kernel()
  {
    return kernel;
  }

  const KernelFunctionType<spacedim, DeviceNumberType<KernelNumberType>> &
  get_kernel() const
  {
    return kernel;
  }

  ClusterTree<spacedim, real_type> &
  get_cluster_tree_trial_space()
  {
    return trial_space.get_cluster_tree();
  }

  const ClusterTree<spacedim, real_type> &
  get_cluster_tree_trial_space() const
  {
    return trial_space.get_cluster_tree();
  }

  ClusterTree<spacedim, real_type> &
  get_cluster_tree_test_space()
  {
    return test_space.get_cluster_tree();
  }

  const ClusterTree<spacedim, real_type> &
  get_cluster_tree_test_space() const
  {
    return test_space.get_cluster_tree();
  }

  BlockClusterTree<spacedim, real_type> &
  get_block_cluster_tree()
  {
    return *block_cluster_tree;
  }

  const BlockClusterTree<spacedim, real_type> &
  get_block_cluster_tree() const
  {
    return *block_cluster_tree;
  }

  const std::vector<types::global_cell_index> &
  get_global_to_local_cell_index_map() const
  {
    return global_to_local_cell_index_map;
  }

  std::vector<types::global_cell_index> &
  get_global_to_local_cell_index_map()
  {
    return global_to_local_cell_index_map;
  }

  const std::vector<types::global_cell_index> &
  get_local_to_global_cell_index_map() const
  {
    return local_to_global_cell_index_map;
  }

  std::vector<types::global_cell_index> &
  get_local_to_global_cell_index_map()
  {
    return local_to_global_cell_index_map;
  }

  const BEMFunctionSpace<dim,
                         spacedim,
                         SearchableMaterialIdContainer,
                         real_type> &
  get_trial_space() const
  {
    return trial_space;
  }

  const BEMFunctionSpace<dim,
                         spacedim,
                         SearchableMaterialIdContainer,
                         real_type> &
  get_test_space() const
  {
    return test_space;
  }

  const std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *> &
  get_cell_iterator_ptrs() const
  {
    return cell_iterator_ptrs;
  }

  std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *> &
  get_cell_iterator_ptrs()
  {
    return cell_iterator_ptrs;
  }

private:
  KernelFunctionType<spacedim, DeviceNumberType<KernelNumberType>> kernel;
  const BEMFunctionSpace<dim,
                         spacedim,
                         SearchableMaterialIdContainer,
                         real_type>                               &trial_space;
  const BEMFunctionSpace<dim,
                         spacedim,
                         SearchableMaterialIdContainer,
                         real_type>                               &test_space;
  /**
   * Whether the bilinear form is symmetric.
   *
   * Only when the kernel function of the boundary integral operator is
   * symmetric and the trial and test spaces are the same BEM function
   * space, the bilinear form is symmetric.
   */
  bool                                                   is_symmetric;
  std::unique_ptr<BlockClusterTree<spacedim, real_type>> block_cluster_tree;
  /**
   * Total number of cells used by the trial and test function spaces.
   */
  types::global_cell_index n_cells;
  /**
   * Map from global cell indices to local cell indices.
   */
  std::vector<types::global_cell_index> global_to_local_cell_index_map;
  /**
   * Map from local cell indices to global cell indices.
   */
  std::vector<types::global_cell_index> local_to_global_cell_index_map;
  /**
   * A list of cell iterator pointers which correspond to the list of used
   * cells.
   *
   * N.B. The actual iterators are managed by BEM function spaces of a BEM
   * bilinear form.
   */
  std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>
    cell_iterator_ptrs;
};


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
BEMBilinearForm<dim,
                spacedim,
                SearchableMaterialIdContainer,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  BEMBilinearForm(const BEMFunctionSpace<dim,
                                         spacedim,
                                         SearchableMaterialIdContainer,
                                         real_type> &trial_space_,
                  const BEMFunctionSpace<dim,
                                         spacedim,
                                         SearchableMaterialIdContainer,
                                         real_type> &test_space_)
  : kernel()
  , trial_space(trial_space_)
  , test_space(test_space_)
  // We make a predicate about if the trial and test spaces are the same BEM
  // function space by checking the equality of their memory addresses.
  , is_symmetric(kernel.is_symmetric() && (&trial_space == &test_space))
  , n_cells(0)
{
  const types::global_cell_index n_cells_in_tria =
    trial_space.get_dof_handler().get_triangulation().n_active_cells();
  // At the moment, the trial space and the test space are constructed on the
  // same triangulation, hence their associated numbers of cells should be the
  // same.
  AssertDimension(
    n_cells_in_tria,
    test_space.get_dof_handler().get_triangulation().n_active_cells());

  // If one of the function spaces (test or trial space) is constructed on the
  // full domain, i.e. the complete triangulation, all cells are used. Then we
  // directly set the global-to-local and local-to-global cell index maps to a
  // linear range starting from 0 with a step 1.
  if (trial_space.get_is_full_domain() || test_space.get_is_full_domain())
    {
      global_to_local_cell_index_map.resize(n_cells_in_tria);
      gen_linear_indices<vector_uta, types::global_cell_index>(
        global_to_local_cell_index_map);
      local_to_global_cell_index_map = global_to_local_cell_index_map;
      cell_iterator_ptrs.resize(n_cells_in_tria);
      n_cells = n_cells_in_tria;

      types::global_cell_index c = 0;
      if (trial_space.get_is_full_domain())
        {
          for (auto &cell_iter : trial_space.get_cell_iterators())
            {
              cell_iterator_ptrs[c] = &cell_iter;
              c++;
            }
        }
      else
        {
          for (auto &cell_iter : test_space.get_cell_iterators())
            {
              cell_iterator_ptrs[c] = &cell_iter;
              c++;
            }
        }
    }
  else
    {
      // Initialize all entries in the global-to-local cell index map to the
      // total number of cells in the triangulation, which indicates they have
      // not been touched yet.
      global_to_local_cell_index_map.assign(n_cells_in_tria, n_cells_in_tria);
      local_to_global_cell_index_map.reserve(n_cells_in_tria);
      cell_iterator_ptrs.reserve(n_cells_in_tria);

      // Collect cells in the trial space first.
      n_cells = BEMTools::generate_maps_between_global_and_local_cell_indices(
        0,
        trial_space.get_dof_to_cell_topo(),
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs);

      // When the trial and test spaces are not the same BEM function space, we
      // should also collect cells in the test space.
      if ((&trial_space != &test_space))
        n_cells = BEMTools::generate_maps_between_global_and_local_cell_indices(
          n_cells,
          test_space.get_dof_to_cell_topo(),
          global_to_local_cell_index_map,
          local_to_global_cell_index_map,
          cell_iterator_ptrs);

      local_to_global_cell_index_map.shrink_to_fit();
      cell_iterator_ptrs.shrink_to_fit();
    }
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
void
BEMBilinearForm<
  dim,
  spacedim,
  SearchableMaterialIdContainer,
  KernelFunctionType,
  RangeNumberType,
  KernelNumberType>::build_block_cluster_tree(const real_type    eta,
                                              const unsigned int n_min,
                                              const unsigned int cutoff_level)
{
  // When building a block cluster tree, the test space appears before the trial
  // space, since the test space is related to matrix rows, while the trial
  // space is related to matrix columns.
  block_cluster_tree = std::make_unique<BlockClusterTree<spacedim, real_type>>(
    test_space.get_cluster_tree(), trial_space.get_cluster_tree(), eta, n_min);
  block_cluster_tree->partition(cutoff_level);
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
BEMBilinearForm<dim,
                spacedim,
                SearchableMaterialIdContainer,
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
    const Table<2, Point<spacedim, real_type>> &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<2, Point<spacedim, real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    SubdomainTopology<dim, spacedim>               &subdomain_topology)
{
  // The kernel does not need regularization.
  Assert(!kernel.needs_regularization(), ExcInternalError());

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

  if (hmat_params.cpu_serial_without_producer_consumer)
    {
      fill_hmatrix_with_aca_plus_serial<dim,
                                        spacedim,
                                        KernelFunctionType,
                                        RangeNumberType,
                                        KernelNumberType,
                                        SurfaceNormalDetector<dim, spacedim>>(
        *hmat,
        hmat_params,
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
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs,
        tria_mapping_support_points_cpu,
        tria_mapping_support_points_gpu,
        tria_mapping_indices_gpu,
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        is_symmetric);
    }
  else
    {
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
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs,
        tria_mapping_support_points_cpu,
        tria_mapping_support_points_gpu,
        tria_mapping_indices_gpu,
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        is_symmetric);
    }

  return hmat;
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
BEMBilinearForm<dim,
                spacedim,
                SearchableMaterialIdContainer,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  build_hmatrix_with_mass_matrix(
    const ConfHMatrix                       &hmat_params,
    const ConfSauterQuadNearField           &sauter_quad_near_field_params,
    const ConfSauterQuadFarField            &sauter_quad_far_field_params,
    const ConfParallelization               &parallel_params,
    const DeviceNumberType<KernelNumberType> kernel_factor,
    const real_type                          mass_matrix_factor,
    const SauterQuadratureRule<dim>         &sauter_quad_rule,
    const QGauss<dim>                       &mass_matrix_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                               &material_id_to_mapping_index,
    const Table<2, Point<spacedim, real_type>> &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<2, Point<spacedim, real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    SubdomainTopology<dim, spacedim>               &subdomain_topology)
{
  // The kernel does not need regularization.
  Assert(!kernel.needs_regularization(), ExcInternalError());

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

  if (hmat_params.cpu_serial_without_producer_consumer)
    {
      fill_hmatrix_with_aca_plus_serial<dim,
                                        spacedim,
                                        KernelFunctionType,
                                        RangeNumberType,
                                        KernelNumberType,
                                        SurfaceNormalDetector<dim, spacedim>>(
        *hmat,
        hmat_params,
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
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs,
        tria_mapping_support_points_cpu,
        tria_mapping_support_points_gpu,
        tria_mapping_indices_gpu,
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        is_symmetric);
    }
  else
    {
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
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs,
        tria_mapping_support_points_cpu,
        tria_mapping_support_points_gpu,
        tria_mapping_indices_gpu,
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        is_symmetric);
    }

  return hmat;
}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
BEMBilinearForm<dim,
                spacedim,
                SearchableMaterialIdContainer,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::
  build_hmatrix_with_regularization(
    const ConfHMatrix                           &hmat_params,
    const ConfSauterQuadNearField               &sauter_quad_near_field_params,
    const ConfSauterQuadFarField                &sauter_quad_far_field_params,
    const ConfParallelization                   &parallel_params,
    const DeviceNumberType<KernelNumberType>     kernel_factor,
    const std::vector<Vector<KernelNumberType>> &mass_vmult_weq,
    const KernelNumberType                       stabilization_factor,
    const SauterQuadratureRule<dim>             &sauter_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                               &material_id_to_mapping_index,
    const Table<2, Point<spacedim, real_type>> &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<2, Point<spacedim, real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    SubdomainTopology<dim, spacedim>               &subdomain_topology)
{
  // The kernel must be regularized, which involves computation of surface
  // curls.
  Assert(kernel.needs_regularization(), ExcInternalError());
  // When the stabilization vectors @p mass_vmult_weq is not empty, the kernel
  // needs stabilization.
  if (mass_vmult_weq.size() > 0)
    Assert(kernel.needs_stabilization_on_full_domain(), ExcInternalError());

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

  if (hmat_params.cpu_serial_without_producer_consumer)
    {
      fill_hmatrix_with_aca_plus_serial<dim,
                                        spacedim,
                                        KernelFunctionType,
                                        RangeNumberType,
                                        KernelNumberType,
                                        SurfaceNormalDetector<dim, spacedim>>(
        *hmat,
        hmat_params,
        kernel,
        kernel_factor,
        mass_vmult_weq,
        stabilization_factor,
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
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs,
        tria_mapping_support_points_cpu,
        tria_mapping_support_points_gpu,
        tria_mapping_indices_gpu,
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        is_symmetric);
    }
  else
    {
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
        mass_vmult_weq,
        stabilization_factor,
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
        global_to_local_cell_index_map,
        local_to_global_cell_index_map,
        cell_iterator_ptrs,
        tria_mapping_support_points_cpu,
        tria_mapping_support_points_gpu,
        tria_mapping_indices_gpu,
        SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
        is_symmetric);
    }

  return hmat;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_BEM_BILINEAR_FORM_H_
