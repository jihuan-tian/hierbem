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
 * @file preconditioner_for_helmholtz_acoustic_dirichlet.h
 * @brief Operator preconditioner used for the Helmholtz acoustic equation with
 * Dirichlet boundary condition.
 *
 * @date 2025-12-17
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_HELMHOLTZ_ACOUSTIC_DIRICHLET_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_HELMHOLTZ_ACOUSTIC_DIRICHLET_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/table.h>
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
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "platform_shared/helmholtz_acoustic_kernels.h"
#include "preconditioners/operator_preconditioner.h"
#include "preconditioners/preconditioner_for_single_layer_bio.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * @brief Class for the dual mesh operator preconditioner, which preconditions
 * the single layer potential boundary integral operator in the Helmholtz
 * acoustic equation.
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
class HelmholtzAcousticSingleLayerPreconditioner
  : public SingleLayerPreconditioner<dim, spacedim, RangeNumberType>
{
public:
  /**
   * Constructor for the preconditioner on the full domain.
   */
  HelmholtzAcousticSingleLayerPreconditioner(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const ConfOperatorPreconditioner           &op_precond_params);

  /**
   * Constructor for the preconditioner on a subdomain.
   */
  HelmholtzAcousticSingleLayerPreconditioner(
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
   *
   * @param sauter_quad_rule1 Sauter quadrature rule for the first part of the
   * matrix, which involves surface curl.
   * @param sauter_quad_rule2 Sauter quadrature rule for the second part of the
   * matrix, the kernel of which is equivalent to the single layer potential
   * kernel.
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
    const Table<
      2,
      Point<spacedim,
            typename numbers::NumberTraits<KernelNumberType>::real_type>>
      &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<
      2,
      Point<spacedim,
            typename numbers::NumberTraits<KernelNumberType>::real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    const SurfaceNormalDetector                    &normal_detector,
    const SauterQuadratureRule<dim>                &sauter_quad_rule1,
    const SauterQuadratureRule<dim>                &sauter_quad_rule2,
    const Quadrature<dim>                          &quad_rule_for_mass);

  /**
   * Compute the product of the preconditioner H-matrix on the refined mesh and
   * an input vector.
   *
   * Because the regularized H-matrix for the hyper singular boundary integral
   * operator is split into two parts, two matrix/vector multiplications are
   * performed.
   *
   * This virtual function is called by the base class member function
   * <tt>OperatorPreconditioner::vmult</tt>.
   */
  void
  vmult_preconditioner_hmat(Vector<RangeNumberType>       &y,
                            const Vector<RangeNumberType> &x) const override;

  void
  set_kappa(const DeviceNumberType<KernelNumberType> kappa_)
  {
    hyper_singular_kernel1.set_kappa(kappa_);
    hyper_singular_kernel2.set_kappa(kappa_);
  }

private:
  /**
   * Kernel function for the first part of the regularized hyper singular
   * boundary integral operator in the Helmholtz acoustic equation.
   *
   * \alert{Because this kernel function is to be evaluated on the device, its
   * number type, i.e. the second template parameter should be
   * <tt>DeviceNumberType<KernelNumberType></tt>.}
   */
  HierBEM::PlatformShared::HelmholtzAcousticKernel::
    HyperSingularKernelRegular1<spacedim, DeviceNumberType<KernelNumberType>>
      hyper_singular_kernel1;
  /**
   * Kernel function for the second part of the regularized hyper singular
   * boundary integral operator in the Helmholtz acoustic equation.
   */
  HierBEM::PlatformShared::HelmholtzAcousticKernel::
    HyperSingularKernelRegular2<spacedim, DeviceNumberType<KernelNumberType>>
      hyper_singular_kernel2;
  /**
   * The Galerkin matrix for the second part of the regularized preconditioner.
   *
   * This part will be added into the first part of the matrix, which has been
   * defined in the base class <tt>OperatorPreconditioner</tt>.
   */
  HMatrix<spacedim, RangeNumberType> preconditioner_hmat2;
};


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
HelmholtzAcousticSingleLayerPreconditioner<dim,
                                           spacedim,
                                           RangeNumberType,
                                           KernelNumberType>::
  HelmholtzAcousticSingleLayerPreconditioner(
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
HelmholtzAcousticSingleLayerPreconditioner<dim,
                                           spacedim,
                                           RangeNumberType,
                                           KernelNumberType>::
  HelmholtzAcousticSingleLayerPreconditioner(
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
HelmholtzAcousticSingleLayerPreconditioner<dim,
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
    const Table<
      2,
      Point<spacedim,
            typename numbers::NumberTraits<KernelNumberType>::real_type>>
      &tria_mapping_support_points_cpu,
    const CUDAWrappers::CUDATable<
      2,
      Point<spacedim,
            typename numbers::NumberTraits<KernelNumberType>::real_type>>
      &tria_mapping_support_points_gpu,
    const CUDAWrappers::CUDATable<1, unsigned int> &tria_mapping_indices_gpu,
    const SurfaceNormalDetector                    &normal_detector,
    const SauterQuadratureRule<dim>                &sauter_quad_rule1,
    const SauterQuadratureRule<dim>                &sauter_quad_rule2,
    const Quadrature<dim>                          &quad_rule_for_mass)
{
  OperatorPreconditioner<dim, spacedim, RangeNumberType>::setup_preconditioner(
    hmat_params, mappings, quad_rule_for_mass);

  // Build the H-matrix for the first part of the regularized bilinear form for
  // the hyper singular boundary integral operator.
  this->template build_preconditioner_hmat_on_refined_mesh<
    HierBEM::PlatformShared::HelmholtzAcousticKernel::
      HyperSingularKernelRegular1,
    KernelNumberType>(this->preconditioner_hmat,
                      hmat_params,
                      sauter_quad_near_field_params,
                      sauter_quad_far_field_params,
                      parallel_params,
                      hyper_singular_kernel1,
                      subdomain_topology,
                      mappings,
                      material_id_to_mapping_index,
                      tria_mapping_support_points_cpu,
                      tria_mapping_support_points_gpu,
                      tria_mapping_indices_gpu,
                      normal_detector,
                      sauter_quad_rule1);

  // Build the H-matrix for the second part of the regularized bilinear form for
  // the hyper singular boundary integral operator.
  this->template build_preconditioner_hmat_on_refined_mesh<
    HierBEM::PlatformShared::HelmholtzAcousticKernel::
      HyperSingularKernelRegular2,
    KernelNumberType>(preconditioner_hmat2,
                      hmat_params,
                      sauter_quad_near_field_params,
                      sauter_quad_far_field_params,
                      parallel_params,
                      hyper_singular_kernel2,
                      subdomain_topology,
                      mappings,
                      material_id_to_mapping_index,
                      tria_mapping_support_points_cpu,
                      tria_mapping_support_points_gpu,
                      tria_mapping_indices_gpu,
                      normal_detector,
                      sauter_quad_rule2);

  // Add the second part H-matrix into the first part without rank truncation.
  this->preconditioner_hmat.add(preconditioner_hmat2);
  preconditioner_hmat2.release();
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
void
HelmholtzAcousticSingleLayerPreconditioner<dim,
                                           spacedim,
                                           RangeNumberType,
                                           KernelNumberType>::
  vmult_preconditioner_hmat(Vector<RangeNumberType>       &y,
                            const Vector<RangeNumberType> &x) const
{
  this->preconditioner_hmat.vmult(y, x);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_HELMHOLTZ_ACOUSTIC_DIRICHLET_H_
