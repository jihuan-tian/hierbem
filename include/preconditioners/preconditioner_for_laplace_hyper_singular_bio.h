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
 * @file preconditioner_for_laplace_hyper_singular_bio.h
 * @brief Dual mesh operator preconditioner for the hyper singular boundary
 * integral operator in the Laplace equation.
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_HYPER_SINGULAR_BIO_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_HYPER_SINGULAR_BIO_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
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
#include "dofs/dof_to_cell_topology.h"
#include "dofs/dof_tools_ext.h"
#include "hmatrix/hmatrix_parameters.h"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "preconditioners/operator_preconditioner.h"
#include "preconditioners/preconditioner_for_hyper_singular_bio.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * @brief Class for the dual mesh operator preconditioner, which preconditions
 * the hyper singular boundary integral operator in the Laplace equation.
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
class LaplaceHyperSingularPreconditioner
  : public HyperSingularPreconditioner<dim, spacedim, RangeNumberType>
{
public:
  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;

  /**
   * Constructor for the preconditioner on the full domain.
   */
  LaplaceHyperSingularPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const unsigned int                          max_iter    = 1000,
    const real_type                             tol         = 1e-8,
    const real_type                             omega       = 1.0,
    const bool                                  log_history = true,
    const bool                                  log_result  = true);

  /**
   * Constructor for the preconditioner on a subdomain.
   */
  LaplaceHyperSingularPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const unsigned int                          max_iter    = 1000,
    const real_type                             tol         = 1e-8,
    const real_type                             omega       = 1.0,
    const bool                                  log_history = true,
    const bool                                  log_result  = true);

  /**
   * Setup the preconditioner by calling the parent class's version as well as
   * building the H-matrix on the refined mesh.
   */
  template <typename SurfaceNormalDetector>
  void
  setup_preconditioner(
    const unsigned int                               thread_num,
    const HMatrixParameters<real_type>              &hmat_params,
    const SubdomainTopology<dim, spacedim>          &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule,
    const Quadrature<dim>           &quad_rule_for_mass);

private:
  /**
   * Kernel function for the single layer potential boundary integral operator.
   *
   * \alert{Because this kernel function is to be evaluated on the device, its
   * number type, i.e. the second template parameter should be
   * <tt>DeviceNumberType<KernelNumberType></tt>.}
   */
  PlatformShared::LaplaceKernel::
    SingleLayerKernel<spacedim, DeviceNumberType<KernelNumberType>>
      slp_kernel;
};


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
LaplaceHyperSingularPreconditioner<dim,
                                   spacedim,
                                   RangeNumberType,
                                   KernelNumberType>::
  LaplaceHyperSingularPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const unsigned int                          max_iter,
    const real_type                             tol,
    const real_type                             omega,
    const bool                                  log_history,
    const bool                                  log_result)
  : HyperSingularPreconditioner<dim, spacedim, RangeNumberType>(
      fe_primal_space,
      fe_dual_space,
      tria,
      primal_space_dof_i2e_numbering,
      primal_space_dof_e2i_numbering,
      max_iter,
      tol,
      omega,
      log_history,
      log_result)
{
  // At the moment, in a Neumann problem, the primal space can only be
  // @p FE_Q(1) and the dual space can only be @p FE_DGQ(0). Therefore, we make
  // assertions here about their numbers of DoFs in a cell and the conformity
  // with continuous function spaces.
  AssertDimension(this->fe_primal_space.dofs_per_cell, 4);
  Assert(this->fe_primal_space.conforms(FiniteElementData<dim>::Conformity::H1),
         ExcInternalError());
  AssertDimension(this->fe_dual_space.dofs_per_cell, 1);
  Assert(this->fe_dual_space.conforms(FiniteElementData<dim>::Conformity::L2),
         ExcInternalError());
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
LaplaceHyperSingularPreconditioner<dim,
                                   spacedim,
                                   RangeNumberType,
                                   KernelNumberType>::
  LaplaceHyperSingularPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const unsigned int                          max_iter,
    const real_type                             tol,
    const real_type                             omega,
    const bool                                  log_history,
    const bool                                  log_result)
  : HyperSingularPreconditioner<dim, spacedim, RangeNumberType>(
      fe_primal_space,
      fe_dual_space,
      tria,
      primal_space_dof_i2e_numbering,
      primal_space_dof_e2i_numbering,
      subdomain_material_ids,
      max_iter,
      tol,
      omega,
      log_history,
      log_result)
{
  // At the moment, in a Neumann problem, the primal space can only be
  // @p FE_Q(1) and the dual space can only be @p FE_DGQ(0). Therefore, we make
  // assertions here about their numbers of DoFs in a cell and the conformity
  // with continuous function spaces.
  AssertDimension(this->fe_primal_space.dofs_per_cell, 4);
  Assert(this->fe_primal_space.conforms(FiniteElementData<dim>::Conformity::H1),
         ExcInternalError());
  AssertDimension(this->fe_dual_space.dofs_per_cell, 1);
  Assert(this->fe_dual_space.conforms(FiniteElementData<dim>::Conformity::L2),
         ExcInternalError());
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
template <typename SurfaceNormalDetector>
void
LaplaceHyperSingularPreconditioner<dim,
                                   spacedim,
                                   RangeNumberType,
                                   KernelNumberType>::
  setup_preconditioner(
    const unsigned int                               thread_num,
    const HMatrixParameters<real_type>              &hmat_params,
    const SubdomainTopology<dim, spacedim>          &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule,
    const Quadrature<dim>           &quad_rule_for_mass)
{
  OperatorPreconditioner<dim, spacedim, RangeNumberType>::setup_preconditioner(
    hmat_params, mappings, quad_rule_for_mass);

  this->template build_preconditioner_hmat_on_refined_mesh<
    PlatformShared::LaplaceKernel::SingleLayerKernel,
    KernelNumberType>(this->preconditioner_hmat,
                      hmat_params,
                      slp_kernel,
                      thread_num,
                      subdomain_topology,
                      mappings,
                      material_id_to_mapping_index,
                      normal_detector,
                      sauter_quad_rule);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_HYPER_SINGULAR_BIO_H_
