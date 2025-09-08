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
 * @file preconditioner_for_laplace_neumann.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_NEUMANN_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_NEUMANN_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <vector>

#include "config.h"
#include "dofs/dof_to_cell_topology.h"
#include "dofs/dof_tools_ext.h"
#include "platform_shared/laplace_kernels.h"
#include "preconditioners/operator_preconditioner.h"

HBEM_NS_OPEN

using namespace dealii;

template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
class PreconditionerForLaplaceNeumann
  : public OperatorPreconditioner<
      dim,
      spacedim,
      HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
      RangeNumberType,
      KernelNumberType>
{
public:
  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;

  /**
   * Constructor for preconditioner on full domain.
   */
  PreconditionerForLaplaceNeumann(
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
   * Constructor for preconditioner on subdomain.
   */
  PreconditionerForLaplaceNeumann(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const std::set<types::material_id> &subdomain_complement_material_ids,
    const unsigned int                  max_iter    = 1000,
    const real_type                     tol         = 1e-8,
    const real_type                     omega       = 1.0,
    const bool                          log_history = true,
    const bool                          log_result  = true);

  virtual void
  build_dof_to_cell_topology() final;

  virtual void
  build_coupling_matrix() final;

  virtual void
  build_averaging_matrix() final;

private:
  /**
   * Collection of cell iterators held in the DoF handler for the primal space
   * on the primal mesh.
   */
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
    cell_iterators_primal_space;
  /**
   * DoF-to-cell topology for the primal space on the primal mesh.
   */
  DoFToCellTopology<dim, spacedim> dof_to_cell_topo_primal_space;
};


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
PreconditionerForLaplaceNeumann<dim,
                                spacedim,
                                RangeNumberType,
                                KernelNumberType>::
  PreconditionerForLaplaceNeumann(
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
  : OperatorPreconditioner<
      dim,
      spacedim,
      HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
      RangeNumberType,
      KernelNumberType>("neumann",
                        fe_primal_space,
                        fe_dual_space,
                        tria,
                        primal_space_dof_i2e_numbering,
                        primal_space_dof_e2i_numbering,
                        std::set<types::material_id>(),
                        std::set<types::material_id>(),
                        true,
                        false,
                        false,
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
PreconditionerForLaplaceNeumann<dim,
                                spacedim,
                                RangeNumberType,
                                KernelNumberType>::
  PreconditionerForLaplaceNeumann(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const std::set<types::material_id> &subdomain_complement_material_ids,
    const unsigned int                  max_iter,
    const real_type                     tol,
    const real_type                     omega,
    const bool                          log_history,
    const bool                          log_result)
  : OperatorPreconditioner<
      dim,
      spacedim,
      HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
      RangeNumberType,
      KernelNumberType>("neumann-subdomain",
                        fe_primal_space,
                        fe_dual_space,
                        tria,
                        primal_space_dof_i2e_numbering,
                        primal_space_dof_e2i_numbering,
                        subdomain_material_ids,
                        subdomain_complement_material_ids,
                        false,
                        true,
                        false,
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
void
PreconditionerForLaplaceNeumann<dim,
                                spacedim,
                                RangeNumberType,
                                KernelNumberType>::build_dof_to_cell_topology()
{
  // Call the parent class's function to build the DoF-to-cell topology for the
  // dual space on the refined mesh.
  OperatorPreconditioner<
    dim,
    spacedim,
    HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel,
    RangeNumberType,
    KernelNumberType>::build_dof_to_cell_topology();

  // Build the DoF-to-cell topology for the primal space on the primal mesh.
  cell_iterators_primal_space.reserve(
    this->tria.n_cells(this->primal_mesh_level));
  if (this->is_full_domain)
    {
      for (const auto &cell :
           this->dof_handler_primal_space.mg_cell_iterators_on_level(
             this->primal_mesh_level))
        cell_iterators_primal_space.push_back(cell);

      DoFToolsExt::build_mg_dof_to_cell_topology(dof_to_cell_topo_primal_space,
                                                 cell_iterators_primal_space,
                                                 this->dof_handler_primal_space,
                                                 this->primal_mesh_level);
    }
  else
    {
      if (this->truncate_function_space_dof_support_within_subdomain)
        {
          for (const auto &cell :
               this->dof_handler_primal_space.mg_cell_iterators_on_level(
                 this->primal_mesh_level))
            {
              auto found_iter =
                this->subdomain_material_ids.find(cell->material_id());

              if (found_iter != this->subdomain_material_ids.end())
                cell_iterators_primal_space.push_back(cell);
            }
        }
      else
        {
          for (const auto &cell :
               this->dof_handler_primal_space.mg_cell_iterators_on_level(
                 this->primal_mesh_level))
            cell_iterators_primal_space.push_back(cell);
        }

      DoFToolsExt::build_mg_dof_to_cell_topology(
        dof_to_cell_topo_primal_space,
        cell_iterators_primal_space,
        this->dof_handler_primal_space,
        this->primal_space_dof_selectors_on_primal_mesh,
        this->primal_mesh_level);
    }
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
void
PreconditionerForLaplaceNeumann<dim,
                                spacedim,
                                RangeNumberType,
                                KernelNumberType>::build_coupling_matrix()
{
  // Generate the dynamic sparsity pattern.
  DynamicSparsityPattern dsp(
    this->is_full_domain ?
      this->dof_handler_primal_space.n_dofs(this->primal_mesh_level) :
      this->primal_space_local_to_full_dof_id_map_on_primal_mesh.size(),
    this->is_full_domain ?
      this->dof_handler_primal_space.n_dofs(this->refined_mesh_level) :
      this->primal_space_local_to_full_dof_id_map_on_refined_mesh.size());

  std::vector<types::global_dof_index> dof_indices_in_primal_cell(
    this->fe_primal_space.dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices_in_refined_cell(
    this->fe_primal_space.dofs_per_cell);

  // Iterate over each cell in the primal mesh.
  for (const auto &cell :
       this->dof_handler_primal_space.mg_cell_iterators_on_level(
         this->primal_mesh_level))
    {
      if (!this->is_full_domain)
        {
          auto found_iter =
            this->subdomain_material_ids.find(cell->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      cell->get_mg_dof_indices(dof_indices_in_primal_cell);
      // Iterate over each child iterator of the current cell.
      unsigned int child_index = 0;
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(dof_indices_in_refined_cell);

          if (this->is_full_domain)
            switch (child_index)
              {
                case 0:
                  dsp.add(dof_indices_in_primal_cell[0],
                          dof_indices_in_refined_cell[0]);
                  dsp.add(dof_indices_in_primal_cell[0],
                          dof_indices_in_refined_cell[1]);
                  dsp.add(dof_indices_in_primal_cell[0],
                          dof_indices_in_refined_cell[2]);
                  dsp.add(dof_indices_in_primal_cell[0],
                          dof_indices_in_refined_cell[3]);
                  dsp.add(dof_indices_in_primal_cell[1],
                          dof_indices_in_refined_cell[1]);
                  dsp.add(dof_indices_in_primal_cell[1],
                          dof_indices_in_refined_cell[3]);
                  dsp.add(dof_indices_in_primal_cell[2],
                          dof_indices_in_refined_cell[2]);
                  dsp.add(dof_indices_in_primal_cell[2],
                          dof_indices_in_refined_cell[3]);
                  dsp.add(dof_indices_in_primal_cell[3],
                          dof_indices_in_refined_cell[3]);

                  break;
                case 1:
                  dsp.add(dof_indices_in_primal_cell[1],
                          dof_indices_in_refined_cell[1]);
                  dsp.add(dof_indices_in_primal_cell[1],
                          dof_indices_in_refined_cell[3]);
                  dsp.add(dof_indices_in_primal_cell[3],
                          dof_indices_in_refined_cell[3]);

                  break;
                case 2:
                  dsp.add(dof_indices_in_primal_cell[2],
                          dof_indices_in_refined_cell[2]);
                  dsp.add(dof_indices_in_primal_cell[2],
                          dof_indices_in_refined_cell[3]);
                  dsp.add(dof_indices_in_primal_cell[3],
                          dof_indices_in_refined_cell[3]);

                  break;
                case 3:
                  dsp.add(dof_indices_in_primal_cell[3],
                          dof_indices_in_refined_cell[3]);

                  break;
              }
          else
            switch (child_index)
              {
                case 0:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[0]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[0]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[0]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[1]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[0]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[1]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[2]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[0]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[2]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[0]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[1]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[1]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[1]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[1]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[2]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[2]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[2]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[2]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[3]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  break;
                case 1:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[1]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[1]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[1]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[1]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[3]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  break;
                case 2:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[2]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[2]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[2]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[2]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[3]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  break;
                case 3:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    dsp.add(
                      this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                        [dof_indices_in_primal_cell[3]],
                      this
                        ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                          [dof_indices_in_refined_cell[3]]);

                  break;
              }

          child_index++;
        }
    }

  // Generate the sparsity pattern.
  this->coupling_matrix_sp.copy_from(dsp);

  // Initialize the sparse matrix \f$C_p\f$.
  this->coupling_matrix.reinit(this->coupling_matrix_sp);

  // Fill values into the coupling matrix.
  // Iterate over each cell in the primal mesh.
  for (const auto &cell :
       this->dof_handler_primal_space.mg_cell_iterators_on_level(
         this->primal_mesh_level))
    {
      if (!this->is_full_domain)
        {
          auto found_iter =
            this->subdomain_material_ids.find(cell->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      cell->get_mg_dof_indices(dof_indices_in_primal_cell);
      // Iterate over each child iterator of the current cell.
      unsigned int child_index = 0;
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(dof_indices_in_refined_cell);

          if (this->is_full_domain)
            switch (child_index)
              {
                case 0:
                  this->coupling_matrix.set(dof_indices_in_primal_cell[0],
                                            dof_indices_in_refined_cell[0],
                                            1.0);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[0],
                                            dof_indices_in_refined_cell[1],
                                            0.5);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[0],
                                            dof_indices_in_refined_cell[2],
                                            0.5);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[0],
                                            dof_indices_in_refined_cell[3],
                                            0.25);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[1],
                                            dof_indices_in_refined_cell[1],
                                            0.5);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[1],
                                            dof_indices_in_refined_cell[3],
                                            0.25);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[2],
                                            dof_indices_in_refined_cell[2],
                                            0.5);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[2],
                                            dof_indices_in_refined_cell[3],
                                            0.25);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[3],
                                            dof_indices_in_refined_cell[3],
                                            0.25);

                  break;
                case 1:
                  this->coupling_matrix.set(dof_indices_in_primal_cell[1],
                                            dof_indices_in_refined_cell[1],
                                            1.0);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[1],
                                            dof_indices_in_refined_cell[3],
                                            0.5);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[3],
                                            dof_indices_in_refined_cell[3],
                                            0.5);

                  break;
                case 2:
                  this->coupling_matrix.set(dof_indices_in_primal_cell[2],
                                            dof_indices_in_refined_cell[2],
                                            1.0);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[2],
                                            dof_indices_in_refined_cell[3],
                                            0.5);
                  this->coupling_matrix.set(dof_indices_in_primal_cell[3],
                                            dof_indices_in_refined_cell[3],
                                            0.5);

                  break;
                case 3:
                  this->coupling_matrix.set(dof_indices_in_primal_cell[3],
                                            dof_indices_in_refined_cell[3],
                                            1.0);

                  break;
              }
          else
            switch (child_index)
              {
                case 0:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[0]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[0]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[0]],
                        1.0);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[1]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[0]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[1]],
                        0.5);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[2]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[0]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[2]],
                        0.5);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[0]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[0]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.25);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[1]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[1]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[1]],
                        0.5);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[1]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.25);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[2]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[2]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[2]],
                        0.5);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[2]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.25);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[3]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.25);

                  break;
                case 1:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[1]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[1]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[1]],
                        1.0);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[1]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[1]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.5);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[3]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.5);

                  break;
                case 2:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[2]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[2]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[2]],
                        1.0);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[2]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[2]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.5);

                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[3]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        0.5);

                  break;
                case 3:
                  if ((this->primal_space_dof_selectors_on_primal_mesh
                         [dof_indices_in_primal_cell[3]]) &&
                      (this->primal_space_dof_selectors_on_refined_mesh
                         [dof_indices_in_refined_cell[3]]))
                    (this->coupling_matrix)
                      .set(
                        this
                          ->primal_space_full_to_local_dof_id_map_on_primal_mesh
                            [dof_indices_in_primal_cell[3]],
                        this
                          ->primal_space_full_to_local_dof_id_map_on_refined_mesh
                            [dof_indices_in_refined_cell[3]],
                        1.0);

                  break;
              }

          child_index++;
        }
    }
}


template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
void
PreconditionerForLaplaceNeumann<dim,
                                spacedim,
                                RangeNumberType,
                                KernelNumberType>::build_averaging_matrix()
{
  // Generate the dynamic sparsity pattern. N.B. The number of DoFs in the
  // primal space on the primal mesh should be the same as the number of DoFs in
  // the dual space on the dual mesh. Therefore, we use this size as the number
  // of rows in \f$C_d\f$.
  DynamicSparsityPattern dsp(
    this->is_full_domain ?
      this->dof_handler_primal_space.n_dofs(this->primal_mesh_level) :
      this->primal_space_local_to_full_dof_id_map_on_primal_mesh.size(),
    this->is_full_domain ?
      this->dof_handler_dual_space.n_dofs(this->refined_mesh_level) :
      this->dual_space_local_to_full_dof_id_map_on_refined_mesh.size());

  // Iterate over each cell in the primal space on the primal mesh.
  auto primal_cell_in_primal_space =
    this->dof_handler_primal_space.begin_mg(this->primal_mesh_level);
  // Because the DoF numbering adopted for the dual space on the dual mesh can
  // be arbitrary, we use the DoF indices held in the DoF handler for the primal
  // space on the primal mesh.
  std::vector<types::global_dof_index> cell_dof_indices_in_primal_space(
    this->fe_primal_space.dofs_per_cell);

  // We need the primal cell in the dual space to traverse its four children in
  // the refined mesh.
  auto primal_cell_in_dual_space =
    this->dof_handler_dual_space.begin_mg(this->primal_mesh_level);
  std::vector<types::global_dof_index> cell_dof_indices_in_dual_space(
    this->fe_dual_space.dofs_per_cell);
  for (; primal_cell_in_primal_space !=
         this->dof_handler_primal_space.end_mg(this->primal_mesh_level);
       primal_cell_in_primal_space++, primal_cell_in_dual_space++)
    {
      // We are relying on the fact that for a given triangulation, different
      // DoF handlers have the same cell order, therefore the two cells here
      // should have the same cell index.
      Assert(primal_cell_in_primal_space->index() ==
               primal_cell_in_dual_space->index(),
             ExcInternalError());

      if (!this->is_full_domain)
        {
          auto found_iter = this->subdomain_material_ids.find(
            primal_cell_in_primal_space->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      primal_cell_in_primal_space->get_mg_dof_indices(
        cell_dof_indices_in_primal_space);

      // Iterate over each child of the current primal cell in the dual space.
      unsigned int child_index = 0;
      for (const auto &child : primal_cell_in_dual_space->child_iterators())
        {
          child->get_mg_dof_indices(cell_dof_indices_in_dual_space);

          if (this->is_full_domain)
            dsp.add(cell_dof_indices_in_primal_space[child_index],
                    cell_dof_indices_in_dual_space[0]);
          else if ((this->primal_space_dof_selectors_on_primal_mesh
                      [cell_dof_indices_in_primal_space[child_index]]) &&
                   (this->dual_space_dof_selectors_on_refined_mesh
                      [cell_dof_indices_in_dual_space[0]]))
            dsp.add(this->primal_space_full_to_local_dof_id_map_on_primal_mesh
                      [cell_dof_indices_in_primal_space[child_index]],
                    this->dual_space_full_to_local_dof_id_map_on_refined_mesh
                      [cell_dof_indices_in_dual_space[0]]);

          child_index++;
        }
    }

  // Generate the sparsity pattern.
  this->averaging_matrix_sp.copy_from(dsp);

  // Initialize the sparse matrix \f$C_d\f$.
  this->averaging_matrix.reinit(this->averaging_matrix_sp);

  // Fill values into the averaging matrix.

  primal_cell_in_primal_space =
    this->dof_handler_primal_space.begin_mg(this->primal_mesh_level);
  primal_cell_in_dual_space =
    this->dof_handler_dual_space.begin_mg(this->primal_mesh_level);
  for (; primal_cell_in_primal_space !=
         this->dof_handler_primal_space.end_mg(this->primal_mesh_level);
       primal_cell_in_primal_space++, primal_cell_in_dual_space++)
    {
      if (!this->is_full_domain)
        {
          auto found_iter = this->subdomain_material_ids.find(
            primal_cell_in_primal_space->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      primal_cell_in_primal_space->get_mg_dof_indices(
        cell_dof_indices_in_primal_space);

      // Iterate over each child of the current primal cell in the dual space.
      unsigned int child_index = 0;
      for (const auto &child : primal_cell_in_dual_space->child_iterators())
        {
          child->get_mg_dof_indices(cell_dof_indices_in_dual_space);

          if (this->is_full_domain)
            {
              unsigned int number_of_cells_sharing_dof =
                dof_to_cell_topo_primal_space
                  .topology[cell_dof_indices_in_primal_space[child_index]]
                  .size();

              this->averaging_matrix.set(
                cell_dof_indices_in_primal_space[child_index],
                cell_dof_indices_in_dual_space[0],
                1.0 / number_of_cells_sharing_dof);
            }
          else if ((this->primal_space_dof_selectors_on_primal_mesh
                      [cell_dof_indices_in_primal_space[child_index]]) &&
                   (this->dual_space_dof_selectors_on_refined_mesh
                      [cell_dof_indices_in_dual_space[0]]))
            {
              unsigned int number_of_cells_sharing_dof =
                dof_to_cell_topo_primal_space
                  .topology[cell_dof_indices_in_primal_space[child_index]]
                  .size();

              this->averaging_matrix.set(
                (this->primal_space_full_to_local_dof_id_map_on_primal_mesh)
                  [cell_dof_indices_in_primal_space[child_index]],
                (this->dual_space_full_to_local_dof_id_map_on_refined_mesh)
                  [cell_dof_indices_in_dual_space[0]],
                1.0 / number_of_cells_sharing_dof);
            }

          child_index++;
        }
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_NEUMANN_H_
