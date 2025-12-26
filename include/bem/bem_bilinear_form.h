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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/vector.h>

#include <map>
#include <memory>
#include <vector>

#include "bem/bem_function_space.h"
#include "cad_mesh/subdomain_topology.h"
#include "cluster_tree/block_cluster_tree.h"
#include "config.h"
#include "hmatrix/aca_plus/aca_config.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_support.h"
#include "mapping/mapping_info.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/number_traits.h"

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
   */
  void
  build_block_cluster_tree(const double eta, const unsigned int n_min);

  /**
   * Build an H-matrix for the bilinear form.
   */
  std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
  build_hmatrix(const unsigned int                       thread_num,
                const unsigned int                       max_rank,
                const real_type                          epsilon,
                const DeviceNumberType<KernelNumberType> kernel_factor,
                const SauterQuadratureRule<dim>         &sauter_quad_rule,
                const std::vector<MappingInfo<dim, spacedim> *> &mappings,
                const std::map<types::material_id, unsigned int>
                                                 &material_id_to_mapping_index,
                SubdomainTopology<dim, spacedim> &subdomain_topology);

  /**
   * Build an H-matrix for the bilinear form and add a mass matrix directly into
   * it.
   */
  std::unique_ptr<HMatrix<spacedim, RangeNumberType>>
  build_hmatrix_with_mass_matrix(
    const unsigned int                               thread_num,
    const unsigned int                               max_rank,
    const real_type                                  epsilon,
    const DeviceNumberType<KernelNumberType>         kernel_factor,
    const real_type                                  mass_matrix_factor,
    const SauterQuadratureRule<dim>                 &sauter_quad_rule,
    const QGauss<dim>                               &mass_matrix_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                     &material_id_to_mapping_index,
    SubdomainTopology<dim, spacedim> &subdomain_topology);

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
    const unsigned int                               thread_num,
    const unsigned int                               max_rank,
    const real_type                                  epsilon,
    const DeviceNumberType<KernelNumberType>         kernel_factor,
    const std::vector<Vector<KernelNumberType>>     &mass_vmult_weq,
    const KernelNumberType                           stabilization_factor,
    const SauterQuadratureRule<dim>                 &sauter_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                     &material_id_to_mapping_index,
    SubdomainTopology<dim, spacedim> &subdomain_topology);

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
  // Whether the bilinear form is symmetric, i.e. the trial space is the same as
  // the test space.
  bool                                                   is_symmetric;
  std::unique_ptr<BlockClusterTree<spacedim, real_type>> block_cluster_tree;
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
  , is_symmetric(kernel.is_symmetric() && (&trial_space == &test_space))
{}


template <int dim,
          int spacedim,
          typename SearchableMaterialIdContainer,
          template <int, typename>
          typename KernelFunctionType,
          typename RangeNumberType,
          typename KernelNumberType>
void
BEMBilinearForm<dim,
                spacedim,
                SearchableMaterialIdContainer,
                KernelFunctionType,
                RangeNumberType,
                KernelNumberType>::build_block_cluster_tree(const double eta,
                                                            const unsigned int
                                                              n_min)
{
  // When building a block cluster tree, the test space appears before the trial
  // space, since the test space is related to matrix rows, while the trial
  // space is related to matrix columns.
  block_cluster_tree = std::make_unique<BlockClusterTree<spacedim, real_type>>(
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
  build_hmatrix(const unsigned int                       thread_num,
                const unsigned int                       max_rank,
                const real_type                          epsilon,
                const DeviceNumberType<KernelNumberType> kernel_factor,
                const SauterQuadratureRule<dim>         &sauter_quad_rule,
                const std::vector<MappingInfo<dim, spacedim> *> &mappings,
                const std::map<types::material_id, unsigned int>
                                                 &material_id_to_mapping_index,
                SubdomainTopology<dim, spacedim> &subdomain_topology)
{
  // The kernel does not need regularization.
  Assert(!kernel.needs_regularization(), ExcInternalError());

  HMatrixSupport::Property  property = is_symmetric ?
                                         HMatrixSupport::Property::symmetric :
                                         HMatrixSupport::Property::general;
  HMatrixSupport::BlockType block_type =
    HMatrixSupport::BlockType::diagonal_block;
  auto hmat = std::make_unique<HMatrix<spacedim, RangeNumberType>>(
    *block_cluster_tree, max_rank, property, block_type);

  fill_hmatrix_with_aca_plus_smp<dim,
                                 spacedim,
                                 KernelFunctionType,
                                 RangeNumberType,
                                 KernelNumberType,
                                 SurfaceNormalDetector<dim, spacedim>>(
    thread_num,
    *hmat,
    ACAConfig<real_type>(max_rank, epsilon, block_cluster_tree->get_eta()),
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
    const unsigned int                               thread_num,
    const unsigned int                               max_rank,
    const real_type                                  epsilon,
    const DeviceNumberType<KernelNumberType>         kernel_factor,
    const real_type                                  mass_matrix_factor,
    const SauterQuadratureRule<dim>                 &sauter_quad_rule,
    const QGauss<dim>                               &mass_matrix_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                     &material_id_to_mapping_index,
    SubdomainTopology<dim, spacedim> &subdomain_topology)
{
  // The kernel does not need regularization.
  Assert(!kernel.needs_regularization(), ExcInternalError());

  HMatrixSupport::Property  property = is_symmetric ?
                                         HMatrixSupport::Property::symmetric :
                                         HMatrixSupport::Property::general;
  HMatrixSupport::BlockType block_type =
    HMatrixSupport::BlockType::diagonal_block;
  auto hmat = std::make_unique<HMatrix<spacedim, RangeNumberType>>(
    *block_cluster_tree, max_rank, property, block_type);

  fill_hmatrix_with_aca_plus_smp<dim,
                                 spacedim,
                                 KernelFunctionType,
                                 RangeNumberType,
                                 KernelNumberType,
                                 SurfaceNormalDetector<dim, spacedim>>(
    thread_num,
    *hmat,
    ACAConfig<real_type>(max_rank, epsilon, block_cluster_tree->get_eta()),
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
    const unsigned int                               thread_num,
    const unsigned int                               max_rank,
    const real_type                                  epsilon,
    const DeviceNumberType<KernelNumberType>         kernel_factor,
    const std::vector<Vector<KernelNumberType>>     &mass_vmult_weq,
    const KernelNumberType                           stabilization_factor,
    const SauterQuadratureRule<dim>                 &sauter_quad_rule,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                     &material_id_to_mapping_index,
    SubdomainTopology<dim, spacedim> &subdomain_topology)
{
  // The kernel must be regularized.
  Assert(kernel.needs_regularization(), ExcInternalError());
  // When the stabilziation vectors @p mass_vmult_weq is not empty, the kernel
  // needs stabilization.
  if (mass_vmult_weq.size() > 0)
    {
      Assert(kernel.needs_stabilization_on_full_domain(), ExcInternalError());
    }

  HMatrixSupport::Property  property = is_symmetric ?
                                         HMatrixSupport::Property::symmetric :
                                         HMatrixSupport::Property::general;
  HMatrixSupport::BlockType block_type =
    HMatrixSupport::BlockType::diagonal_block;
  auto hmat = std::make_unique<HMatrix<spacedim, RangeNumberType>>(
    *block_cluster_tree, max_rank, property, block_type);

  fill_hmatrix_with_aca_plus_smp<dim,
                                 spacedim,
                                 KernelFunctionType,
                                 RangeNumberType,
                                 KernelNumberType,
                                 SurfaceNormalDetector<dim, spacedim>>(
    thread_num,
    *hmat,
    ACAConfig<real_type>(max_rank, epsilon, block_cluster_tree->get_eta()),
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
    SurfaceNormalDetector<dim, spacedim>(subdomain_topology),
    is_symmetric);

  return hmat;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_BEM_BEM_BILINEAR_FORM_H_