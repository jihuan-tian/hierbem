// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file preconditioner_for_laplace_single_layer_bio.h
 * @brief Dual mesh operator preconditioner for the single layer potential
 * boundary integral operator in the Laplace equation.
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_SINGLE_LAYER_BIO_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_SINGLE_LAYER_BIO_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <map>
#include <set>
#include <vector>

#include "cad_mesh/subdomain_topology.h"
#include "config.h"
#include "config_file/config_structs.h"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "preconditioners/operator_preconditioner.h"
#include "preconditioners/preconditioner_for_single_layer_bio.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * @brief Class for the dual mesh operator preconditioner, which preconditions
 * the single layer potential boundary integral operator in the Laplace
 * equation.
 *
 * @tparam dim Manifold dimension of the surface
 * @tparam spacedim Space dimension
 * @tparam RangeNumberType Number type of matrix and vector entries
 * @tparam KernelNumberType Number type of kernel function values on the host
 */
template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
class LaplaceSingleLayerPreconditioner
  : public SingleLayerPreconditioner<dim, spacedim, RangeNumberType>
{
public:
  /**
   * Constructor for the preconditioner on the full domain.
   */
  LaplaceSingleLayerPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const ConfOperatorPreconditioner           &op_precond_params);

  /**
   * Constructor for the preconditioner on a subdomain.
   */
  LaplaceSingleLayerPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const ConfOperatorPreconditioner           &op_precond_params);

  /**
   * Setup the preconditioner by calling the parent class's version as well as
   * building the H-matrix on the refined mesh.
   */
  template <typename SurfaceNormalDetector>
  void
  setup_preconditioner(
    const ConfHMatrix                      &hmat_params,
    const ConfSauterQuadNearField          &sauter_quad_near_field_params,
    const ConfSauterQuadFarField           &sauter_quad_far_field_params,
    const ConfParallelization              &parallel_params,
    const SubdomainTopology<dim, spacedim> &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule,
    const Quadrature<dim>           &quad_rule_for_mass);

private:
  /**
   * Kernel function for the regularized hyper singular boundary integral
   * operator in the Laplace equation.
   *
   * \alert{Because this kernel function is to be evaluated on the device, its
   * number type, i.e. the second template parameter should be
   * <tt>DeviceNumberType<KernelNumberType></tt>.}
   */
  PlatformShared::LaplaceKernel::
    HyperSingularKernelRegular<spacedim, DeviceNumberType<KernelNumberType>>
      hyper_singular_kernel;
};


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
LaplaceSingleLayerPreconditioner<dim,
                                 spacedim,
                                 RangeNumberType,
                                 KernelNumberType>::
  LaplaceSingleLayerPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const ConfOperatorPreconditioner           &op_precond_params)
  : SingleLayerPreconditioner<dim, spacedim, RangeNumberType>(
      fe_primal_space,
      fe_dual_space,
      tria,
      primal_space_dof_i2e_numbering,
      primal_space_dof_e2i_numbering,
      op_precond_params)
{
  // At the moment, in a Dirichlet problem, the primal space can only be
  // @p FE_DGQ(0) and the dual space can only be @p FE_Q(1). Therefore, we make
  // assertions here about their numbers of DoFs in a cell and the conformity
  // with continuous function spaces.
  AssertDimension(this->fe_primal_space.dofs_per_cell, 1);
  Assert(this->fe_primal_space.conforms(FiniteElementData<dim>::Conformity::L2),
         ExcInternalError());
  AssertDimension(this->fe_dual_space.dofs_per_cell, 4);
  Assert(this->fe_dual_space.conforms(FiniteElementData<dim>::Conformity::H1),
         ExcInternalError());
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
LaplaceSingleLayerPreconditioner<dim,
                                 spacedim,
                                 RangeNumberType,
                                 KernelNumberType>::
  LaplaceSingleLayerPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const ConfOperatorPreconditioner           &op_precond_params)
  : SingleLayerPreconditioner<dim, spacedim, RangeNumberType>(
      fe_primal_space,
      fe_dual_space,
      tria,
      primal_space_dof_i2e_numbering,
      primal_space_dof_e2i_numbering,
      subdomain_material_ids,
      op_precond_params)
{
  // At the moment, in a Dirichlet problem, the primal space can only be
  // @p FE_DGQ(0) and the dual space can only be @p FE_Q(1). Therefore, we make
  // assertions here about their numbers of DoFs in a cell and the conformity
  // with continuous function spaces.
  AssertDimension(this->fe_primal_space.dofs_per_cell, 1);
  Assert(this->fe_primal_space.conforms(FiniteElementData<dim>::Conformity::L2),
         ExcInternalError());
  AssertDimension(this->fe_dual_space.dofs_per_cell, 4);
  Assert(this->fe_dual_space.conforms(FiniteElementData<dim>::Conformity::H1),
         ExcInternalError());
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
template <typename SurfaceNormalDetector>
void
LaplaceSingleLayerPreconditioner<dim,
                                 spacedim,
                                 RangeNumberType,
                                 KernelNumberType>::
  setup_preconditioner(
    const ConfHMatrix                      &hmat_params,
    const ConfSauterQuadNearField          &sauter_quad_near_field_params,
    const ConfSauterQuadFarField           &sauter_quad_far_field_params,
    const ConfParallelization              &parallel_params,
    const SubdomainTopology<dim, spacedim> &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule,
    const Quadrature<dim>           &quad_rule_for_mass)
{
  // Call the parent class's function to setup the preconditioner, but without
  // building the H-matrix on the refined mesh.
  OperatorPreconditioner<dim, spacedim, RangeNumberType>::setup_preconditioner(
    hmat_params, mappings, quad_rule_for_mass);

  this->template build_preconditioner_hmat_on_refined_mesh<
    PlatformShared::LaplaceKernel::HyperSingularKernelRegular,
    KernelNumberType>(this->preconditioner_hmat,
                      hmat_params,
                      sauter_quad_near_field_params,
                      sauter_quad_far_field_params,
                      parallel_params,
                      hyper_singular_kernel,
                      subdomain_topology,
                      mappings,
                      material_id_to_mapping_index,
                      normal_detector,
                      sauter_quad_rule);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_SINGLE_LAYER_BIO_H_
