/**
 * @file preconditioner_for_laplace_dirichlet.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef INCLUDE_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_
#define INCLUDE_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <vector>

#include "hmatrix_symm.h"
#include "laplace_kernels.hcu"
#include "preconditioners/operator_preconditioner.h"

namespace HierBEM
{
  using namespace dealii;

  template <int dim, int spacedim, typename RangeNumberType>
  class PreconditionerForLaplaceDirichlet
    : public OperatorPreconditioner<
        dim,
        spacedim,
        HierBEM::CUDAWrappers::LaplaceKernel::
          HyperSingularKernelRegular<spacedim, RangeNumberType>,
        HMatrixSymm<spacedim, RangeNumberType>,
        RangeNumberType>
  {
  public:
    /**
     * Constructor.
     */
    PreconditionerForLaplaceDirichlet(
      FiniteElement<dim, spacedim>       &fe_primal_space,
      FiniteElement<dim, spacedim>       &fe_dual_space,
      const Triangulation<dim, spacedim> &primal_tria);

    virtual void
    build_coupling_matrix() final;

    virtual void
    build_averaging_matrix() final;

    virtual void
    build_preconditioning_hmat_on_refined_mesh() final;
  };


  template <int dim, int spacedim, typename RangeNumberType>
  PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
    PreconditionerForLaplaceDirichlet(
      FiniteElement<dim, spacedim>       &fe_primal_space,
      FiniteElement<dim, spacedim>       &fe_dual_space,
      const Triangulation<dim, spacedim> &primal_tria)
    : OperatorPreconditioner<
        dim,
        spacedim,
        HierBEM::CUDAWrappers::LaplaceKernel::
          HyperSingularKernelRegular<spacedim, RangeNumberType>,
        HMatrixSymm<spacedim, RangeNumberType>,
        RangeNumberType>(fe_primal_space, fe_dual_space, primal_tria)
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  void
  PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
    build_coupling_matrix()
  {
    // Generate the dynamic sparsity pattern.
    DynamicSparsityPattern dsp(this->dof_handler_primal_space.n_dofs(0),
                               this->dof_handler_primal_space.n_dofs(1));

    AssertDimension(this->fe_primal_space.dofs_per_cell, 1);
    std::vector<types::global_dof_index> dof_indices_in_cell(
      this->fe_primal_space.dofs_per_cell);

    // Iterate over each cell in the primal mesh, i.e. on level 0.
    for (const auto &cell :
         this->dof_handler_primal_space.mg_cell_iterators_on_level(0))
      {
        cell->get_mg_dof_indices(dof_indices_in_cell);
        types::global_dof_index dof_index_in_primal_mesh =
          dof_indices_in_cell[0];
        // Iterate over each child iterator of the current cell, i.e. on
        // level 1.
        for (const auto &child : cell->child_iterators())
          {
            child->get_mg_dof_indices(dof_indices_in_cell);
            types::global_dof_index dof_index_in_refined_mesh =
              dof_indices_in_cell[0];

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
        cell->get_mg_dof_indices(dof_indices_in_cell);
        types::global_dof_index dof_index_in_primal_mesh =
          dof_indices_in_cell[0];
        // Iterate over each child iterator of the current cell, i.e. on
        // level 1.
        for (const auto &child : cell->child_iterators())
          {
            child->get_mg_dof_indices(dof_indices_in_cell);
            types::global_dof_index dof_index_in_refined_mesh =
              dof_indices_in_cell[0];

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
  {}


  template <int dim, int spacedim, typename RangeNumberType>
  void
  PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
    build_preconditioning_hmat_on_refined_mesh()
  {}
} // namespace HierBEM

#endif