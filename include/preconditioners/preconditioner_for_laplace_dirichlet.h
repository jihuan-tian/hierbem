/**
 * @file preconditioner_for_laplace_dirichlet.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <vector>

#include "config.h"
#include "platform_shared/laplace_kernels.h"
#include "preconditioners/operator_preconditioner.h"

HBEM_NS_OPEN

using namespace dealii;

template <int dim, int spacedim, typename RangeNumberType>
class PreconditionerForLaplaceDirichlet
  : public OperatorPreconditioner<
      dim,
      spacedim,
      HierBEM::PlatformShared::LaplaceKernel::
        HyperSingularKernelRegular<spacedim, RangeNumberType>,
      RangeNumberType>
{
public:
  /**
   * Constructor for preconditioner on full domain.
   */
  PreconditionerForLaplaceDirichlet(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &primal_tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const unsigned int                          max_iter    = 1000,
    const double                                tol         = 1e-8,
    const double                                omega       = 1.0,
    const bool                                  log_history = true,
    const bool                                  log_result  = true);

  /**
   * Constructor for preconditioner on subdomain.
   */
  PreconditionerForLaplaceDirichlet(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &primal_tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const unsigned int                          max_iter    = 1000,
    const double                                tol         = 1e-8,
    const double                                omega       = 1.0,
    const bool                                  log_history = true,
    const bool                                  log_result  = true);

  virtual void
  build_coupling_matrix() final;

  virtual void
  build_averaging_matrix() final;
};


template <int dim, int spacedim, typename RangeNumberType>
PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
  PreconditionerForLaplaceDirichlet(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &primal_tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const unsigned int                          max_iter,
    const double                                tol,
    const double                                omega,
    const bool                                  log_history,
    const bool                                  log_result)
  : OperatorPreconditioner<
      dim,
      spacedim,
      HierBEM::PlatformShared::LaplaceKernel::
        HyperSingularKernelRegular<spacedim, RangeNumberType>,
      RangeNumberType>(fe_primal_space,
                       fe_dual_space,
                       primal_tria,
                       primal_space_dof_i2e_numbering,
                       primal_space_dof_e2i_numbering,
                       std::set<types::material_id>(),
                       std::set<types::material_id>(),
                       true,
                       false,
                       max_iter,
                       tol,
                       omega,
                       log_history,
                       log_result)
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


template <int dim, int spacedim, typename RangeNumberType>
PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
  PreconditionerForLaplaceDirichlet(
    FiniteElement<dim, spacedim>               &fe_primal_space,
    FiniteElement<dim, spacedim>               &fe_dual_space,
    const Triangulation<dim, spacedim>         &primal_tria,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering,
    const std::set<types::material_id>         &subdomain_material_ids,
    const unsigned int                          max_iter,
    const double                                tol,
    const double                                omega,
    const bool                                  log_history,
    const bool                                  log_result)
  : OperatorPreconditioner<
      dim,
      spacedim,
      HierBEM::PlatformShared::LaplaceKernel::
        HyperSingularKernelRegular<spacedim, RangeNumberType>,
      RangeNumberType>(fe_primal_space,
                       fe_dual_space,
                       primal_tria,
                       primal_space_dof_i2e_numbering,
                       primal_space_dof_e2i_numbering,
                       subdomain_material_ids,
                       std::set<types::material_id>(),
                       false,
                       false,
                       max_iter,
                       tol,
                       omega,
                       log_history,
                       log_result)
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


template <int dim, int spacedim, typename RangeNumberType>
void
PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
  build_coupling_matrix()
{
  // Generate the dynamic sparsity pattern.
  DynamicSparsityPattern dsp(
    this->is_full_domain ?
      this->dof_handler_primal_space.n_dofs(0) :
      this->primal_space_local_to_full_dof_id_map_on_primal_mesh.size(),
    this->is_full_domain ?
      this->dof_handler_primal_space.n_dofs(1) :
      this->primal_space_local_to_full_dof_id_map_on_refined_mesh.size());

  std::vector<types::global_dof_index> dof_indices_in_cell(
    this->fe_primal_space.dofs_per_cell);

  // Iterate over each cell in the primal mesh, i.e. on level 0.
  for (const auto &cell :
       this->dof_handler_primal_space.mg_cell_iterators_on_level(0))
    {
      if (!this->is_full_domain)
        {
          auto found_iter =
            this->subdomain_material_ids.find(cell->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      cell->get_mg_dof_indices(dof_indices_in_cell);
      if (!this->is_full_domain)
        // Make sure the current DoF is selected.
        Assert(
          this
            ->primal_space_dof_selectors_on_primal_mesh[dof_indices_in_cell[0]],
          ExcInternalError());

      types::global_dof_index dof_index_in_primal_mesh =
        this->is_full_domain ?
          dof_indices_in_cell[0] :
          this->primal_space_full_to_local_dof_id_map_on_primal_mesh
            [dof_indices_in_cell[0]];
      // Iterate over each child iterator of the current cell, i.e. on
      // level 1.
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(dof_indices_in_cell);
          if (!this->is_full_domain)
            // Make sure the current DoF is selected.
            Assert(this->primal_space_dof_selectors_on_refined_mesh
                     [dof_indices_in_cell[0]],
                   ExcInternalError());

          types::global_dof_index dof_index_in_refined_mesh =
            this->is_full_domain ?
              dof_indices_in_cell[0] :
              this->primal_space_full_to_local_dof_id_map_on_refined_mesh
                [dof_indices_in_cell[0]];

          dsp.add(dof_index_in_primal_mesh, dof_index_in_refined_mesh);
        }
    }

  // Generate the sparsity pattern.
  this->coupling_matrix_sp.copy_from(dsp);

  // Initialize the sparse matrix \f$C_p\f$.
  this->coupling_matrix.reinit(this->coupling_matrix_sp);

  // Fill values into the coupling matrix.

  // Iterate over each cell in the primal mesh, i.e. on level 0.
  for (const auto &cell :
       this->dof_handler_primal_space.mg_cell_iterators_on_level(0))
    {
      if (!this->is_full_domain)
        {
          auto found_iter =
            this->subdomain_material_ids.find(cell->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      cell->get_mg_dof_indices(dof_indices_in_cell);
      if (!this->is_full_domain)
        // Make sure the current DoF is selected.
        Assert(
          this
            ->primal_space_dof_selectors_on_primal_mesh[dof_indices_in_cell[0]],
          ExcInternalError());

      types::global_dof_index dof_index_in_primal_mesh =
        this->is_full_domain ?
          dof_indices_in_cell[0] :
          this->primal_space_full_to_local_dof_id_map_on_primal_mesh
            [dof_indices_in_cell[0]];
      // Iterate over each child iterator of the current cell, i.e. on
      // level 1.
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(dof_indices_in_cell);
          if (!this->is_full_domain)
            // Make sure the current DoF is selected.
            Assert(this->primal_space_dof_selectors_on_refined_mesh
                     [dof_indices_in_cell[0]],
                   ExcInternalError());

          types::global_dof_index dof_index_in_refined_mesh =
            this->is_full_domain ?
              dof_indices_in_cell[0] :
              this->primal_space_full_to_local_dof_id_map_on_refined_mesh
                [dof_indices_in_cell[0]];

          this->coupling_matrix.set(dof_index_in_primal_mesh,
                                    dof_index_in_refined_mesh,
                                    1.0);
        }
    }
}


template <int dim, int spacedim, typename RangeNumberType>
void
PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
  build_averaging_matrix()
{
  // Generate the dynamic sparsity pattern. N.B. The row size of this matrix
  // is the number of DoFs in the dual space on the dual mesh, which is the
  // same as the number of primal cells, when the domain is full. When subdomain
  // is considered, it is equivalent to the number of primal space DoFs on the
  // primal mesh, because @p FE_DGQ is used.
  AssertDimension(
    this->primal_space_local_to_full_dof_id_map_on_primal_mesh.size(),
    this->primal_space_dof_i2e_numbering.size());
  DynamicSparsityPattern dsp(
    this->is_full_domain ?
      this->tria.n_cells(0) :
      this->primal_space_local_to_full_dof_id_map_on_primal_mesh.size(),
    this->is_full_domain ?
      this->dof_handler_dual_space.n_dofs(1) :
      this->dual_space_local_to_full_dof_id_map_on_refined_mesh.size());

  // Store the local DoF indices.
  std::vector<types::global_dof_index> dof_indices_in_cell(
    this->fe_dual_space.dofs_per_cell);
  // Store the full DoF indices.
  std::vector<types::global_dof_index> full_dof_indices_in_cell(
    this->fe_dual_space.dofs_per_cell);

  // Iterate over each cell in the primal mesh, which is equivalent to
  // iterating over each node in the dual mesh. When subdomain is considered,
  // only cells in the subdomain are included.
  types::global_dof_index dof_index_in_dual_mesh = 0;
  for (const auto &cell :
       this->dof_handler_dual_space.mg_cell_iterators_on_level(0))
    {
      if (!this->is_full_domain)
        {
          auto found_iter =
            this->subdomain_material_ids.find(cell->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      // Iterate over each child of the current cell.
      unsigned int child_index = 0;
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(full_dof_indices_in_cell);

          if (!this->is_full_domain)
            {
              // When the subdomain is not the full domain, convert the full DoF
              // indices to local.
              for (unsigned int i = 0; i < dof_indices_in_cell.size(); i++)
                {
                  // Make sure the current DoF is selected.
                  Assert(this->dual_space_dof_selectors_on_refined_mesh
                           [full_dof_indices_in_cell[i]],
                         ExcInternalError());

                  dof_indices_in_cell[i] =
                    this->dual_space_full_to_local_dof_id_map_on_refined_mesh
                      [full_dof_indices_in_cell[i]];
                }
            }
          else
            dof_indices_in_cell = full_dof_indices_in_cell;

          switch (child_index)
            {
                case 0: {
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[0]);
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[1]);
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[2]);
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[3]);

                  break;
                }
                case 1: {
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[1]);
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[3]);

                  break;
                }
                case 2: {
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[2]);
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[3]);

                  break;
                }
                case 3: {
                  dsp.add(dof_index_in_dual_mesh, dof_indices_in_cell[3]);

                  break;
                }
                default: {
                  Assert(false, ExcInternalError());
                }
            }

          child_index++;
        }

      dof_index_in_dual_mesh++;
    }

  // Generate the sparsity pattern.
  this->averaging_matrix_sp.copy_from(dsp);

  // Initialize the sparse matrix \f$C_d\f$.
  this->averaging_matrix.reinit(this->averaging_matrix_sp);

  // Fill values into the averaging matrix.

  // Iterate over each cell in the primal mesh, i.e. on level 0. When subdomain
  // is considered, only cells in the subdomain are included.
  dof_index_in_dual_mesh = 0;
  for (const auto &cell :
       this->dof_handler_dual_space.mg_cell_iterators_on_level(0))
    {
      if (!this->is_full_domain)
        {
          auto found_iter =
            this->subdomain_material_ids.find(cell->material_id());
          if (found_iter == this->subdomain_material_ids.end())
            continue;
        }

      // Iterate over each child of the current cell.
      unsigned int child_index = 0;
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(full_dof_indices_in_cell);

          if (!this->is_full_domain)
            {
              // When the subdomain is not the full domain, convert the full DoF
              // indices to local.
              for (unsigned int i = 0; i < dof_indices_in_cell.size(); i++)
                {
                  // Make sure the current DoF is selected.
                  Assert(this->dual_space_dof_selectors_on_refined_mesh
                           [full_dof_indices_in_cell[i]],
                         ExcInternalError());

                  dof_indices_in_cell[i] =
                    this->dual_space_full_to_local_dof_id_map_on_refined_mesh
                      [full_dof_indices_in_cell[i]];
                }
            }
          else
            dof_indices_in_cell = full_dof_indices_in_cell;

          types::global_dof_index dof_index_in_refined_mesh;
          types::global_dof_index full_dof_index_in_refined_mesh;
          unsigned int            number_of_cells_sharing_dof;

          switch (child_index)
            {
                case 0: {
                  // Vertex 0 in the child cell, which is a corner point of
                  // the primal cell.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[0];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[0];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();
                  this->averaging_matrix.set(dof_index_in_dual_mesh,
                                             dof_index_in_refined_mesh,
                                             1.0 / number_of_cells_sharing_dof);

                  // Vertex 1 in the child cell, which is in the middle of a
                  // primal edge.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[1];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[1];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();

                  switch (number_of_cells_sharing_dof)
                    {
                        case 2: {
                          // The DoF support point is in the middle of a
                          // primal edge at the boundary.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     1.0);

                          break;
                        }
                        case 4: {
                          // The DoF support point is in the middle of a
                          // primal edge in the interior.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     0.5);

                          break;
                        }
                        default: {
                          Assert(false, ExcInternalError());
                        }
                    }

                  // Vertex 2 in the child cell, which is in the middle of a
                  // primal edge.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[2];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[2];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();

                  switch (number_of_cells_sharing_dof)
                    {
                        case 2: {
                          // The DoF support point is in the middle of a
                          // primal edge at the boundary.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     1.0);

                          break;
                        }
                        case 4: {
                          // The DoF support point is in the middle of a
                          // primal edge in the interior.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     0.5);

                          break;
                        }
                        default: {
                          Assert(false, ExcInternalError());
                        }
                    }

                  // Vertex 3 in the child cell, which is in the interior of
                  // the primal cell.
                  dof_index_in_refined_mesh = dof_indices_in_cell[3];
                  this->averaging_matrix.set(dof_index_in_dual_mesh,
                                             dof_index_in_refined_mesh,
                                             1);

                  break;
                }
                case 1: {
                  // Vertex 1 in the child cell, which is a corner point of
                  // the primal cell.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[1];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[1];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();
                  this->averaging_matrix.set(dof_index_in_dual_mesh,
                                             dof_index_in_refined_mesh,
                                             1.0 / number_of_cells_sharing_dof);

                  // Vertex 3 in the child cell, which is in the middle of a
                  // primal edge.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[3];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[3];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();

                  switch (number_of_cells_sharing_dof)
                    {
                        case 2: {
                          // The DoF support point is in the middle of a
                          // primal edge at the boundary.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     1.0);

                          break;
                        }
                        case 4: {
                          // The DoF support point is in the middle of a
                          // primal edge in the interior.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     0.5);

                          break;
                        }
                        default: {
                          Assert(false, ExcInternalError());
                        }
                    }

                  break;
                }
                case 2: {
                  // Vertex 2 in the child cell, which is a corner point of
                  // the primal cell.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[2];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[2];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();
                  this->averaging_matrix.set(dof_index_in_dual_mesh,
                                             dof_index_in_refined_mesh,
                                             1.0 / number_of_cells_sharing_dof);

                  // Vertex 3 in the child cell, which is in the middle of a
                  // primal edge.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[3];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[3];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();

                  switch (number_of_cells_sharing_dof)
                    {
                        case 2: {
                          // The DoF support point is in the middle of a
                          // primal edge at the boundary.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     1.0);

                          break;
                        }
                        case 4: {
                          // The DoF support point is in the middle of a
                          // primal edge in the interior.
                          this->averaging_matrix.set(dof_index_in_dual_mesh,
                                                     dof_index_in_refined_mesh,
                                                     0.5);

                          break;
                        }
                        default: {
                          Assert(false, ExcInternalError());
                        }
                    }

                  break;
                }
                case 3: {
                  // Vertex 3 in the child cell, which is a corner point of
                  // the primal cell.
                  dof_index_in_refined_mesh      = dof_indices_in_cell[3];
                  full_dof_index_in_refined_mesh = full_dof_indices_in_cell[3];
                  number_of_cells_sharing_dof =
                    this->dof_to_cell_topo_dual_space
                      .topology[full_dof_index_in_refined_mesh]
                      .size();
                  this->averaging_matrix.set(dof_index_in_dual_mesh,
                                             dof_index_in_refined_mesh,
                                             1.0 / number_of_cells_sharing_dof);

                  break;
                }
                default: {
                  Assert(false, ExcInternalError());
                }
            }

          child_index++;
        }

      dof_index_in_dual_mesh++;
    }
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_
