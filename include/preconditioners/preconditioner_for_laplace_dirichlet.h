/**
 * @file preconditioner_for_laplace_dirichlet.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_PRECONDITIONER_FOR_LAPLACE_DIRICHLET_H_

#include "platform_shared/laplace_kernels.h"
#include "preconditioners/operator_preconditioner.h"

namespace HierBEM
{
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
     * Constructor.
     */
    PreconditionerForLaplaceDirichlet(
      FiniteElement<dim, spacedim>       &fe_primal_space,
      FiniteElement<dim, spacedim>       &fe_dual_space,
      const Triangulation<dim, spacedim> &primal_tria,
      const std::vector<types::global_dof_index>
        &primal_space_dof_i2e_numbering,
      const std::vector<types::global_dof_index>
                        &primal_space_dof_e2i_numbering,
      const unsigned int max_iter    = 1000,
      const double       tol         = 1e-8,
      const double       omega       = 1.0,
      const bool         log_history = true,
      const bool         log_result  = true);

    virtual void
    build_coupling_matrix() final;

    virtual void
    build_averaging_matrix() final;
  };


  template <int dim, int spacedim, typename RangeNumberType>
  PreconditionerForLaplaceDirichlet<dim, spacedim, RangeNumberType>::
    PreconditionerForLaplaceDirichlet(
      FiniteElement<dim, spacedim>       &fe_primal_space,
      FiniteElement<dim, spacedim>       &fe_dual_space,
      const Triangulation<dim, spacedim> &primal_tria,
      const std::vector<types::global_dof_index>
        &primal_space_dof_i2e_numbering,
      const std::vector<types::global_dof_index>
                        &primal_space_dof_e2i_numbering,
      const unsigned int max_iter,
      const double       tol,
      const double       omega,
      const bool         log_history,
      const bool         log_result)
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
                         max_iter,
                         tol,
                         omega,
                         log_history,
                         log_result)
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
  {
    // Generate the dynamic sparsity pattern. N.B. The row size of this matrix
    // is the number of DoFs in the dual space on the dual mesh, which is the
    // same as the number of primal cells.
    DynamicSparsityPattern dsp(this->tria.get_triangulation().n_cells(0),
                               this->dof_handler_dual_space.n_dofs(1));

    std::vector<types::global_dof_index> dof_indices_in_cell(
      this->fe_dual_space.dofs_per_cell);

    // Iterate over each cell in the primal mesh, which is equivalent to
    // iterating over each node in the dual mesh.
    types::global_dof_index dof_index_in_dual_mesh = 0;
    for (const auto &cell :
         this->dof_handler_dual_space.mg_cell_iterators_on_level(0))
      {
        // Iterate over each child of the current cell.
        unsigned int child_index = 0;
        for (const auto &child : cell->child_iterators())
          {
            child->get_mg_dof_indices(dof_indices_in_cell);

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

    // Fill values into the coupling matrix.

    // Iterate over each cell in the primal mesh, i.e. on level 0.
    dof_index_in_dual_mesh = 0;
    for (const auto &cell :
         this->dof_handler_dual_space.mg_cell_iterators_on_level(0))
      {
        // Iterate over each child of the current cell.
        unsigned int child_index = 0;
        for (const auto &child : cell->child_iterators())
          {
            child->get_mg_dof_indices(dof_indices_in_cell);
            types::global_dof_index dof_index_in_refined_mesh;
            unsigned int            number_of_cells_sharing_dof;

            switch (child_index)
              {
                  case 0: {
                    // Vertex 0 in the child cell, which is a corner point of
                    // the primal cell.
                    dof_index_in_refined_mesh = dof_indices_in_cell[0];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();
                    this->averaging_matrix.set(dof_index_in_dual_mesh,
                                               dof_index_in_refined_mesh,
                                               1.0 /
                                                 number_of_cells_sharing_dof);

                    // Vertex 1 in the child cell, which is in the middle of a
                    // primal edge.
                    dof_index_in_refined_mesh = dof_indices_in_cell[1];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();

                    switch (number_of_cells_sharing_dof)
                      {
                          case 2: {
                            // The DoF support point is in the middle of a
                            // primal edge at the boundary.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
                              dof_index_in_refined_mesh,
                              1.0);

                            break;
                          }
                          case 4: {
                            // The DoF support point is in the middle of a
                            // primal edge in the interior.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
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
                    dof_index_in_refined_mesh = dof_indices_in_cell[2];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();

                    switch (number_of_cells_sharing_dof)
                      {
                          case 2: {
                            // The DoF support point is in the middle of a
                            // primal edge at the boundary.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
                              dof_index_in_refined_mesh,
                              1.0);

                            break;
                          }
                          case 4: {
                            // The DoF support point is in the middle of a
                            // primal edge in the interior.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
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
                    dof_index_in_refined_mesh = dof_indices_in_cell[1];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();
                    this->averaging_matrix.set(dof_index_in_dual_mesh,
                                               dof_index_in_refined_mesh,
                                               1.0 /
                                                 number_of_cells_sharing_dof);

                    // Vertex 3 in the child cell, which is in the middle of a
                    // primal edge.
                    dof_index_in_refined_mesh = dof_indices_in_cell[3];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();

                    switch (number_of_cells_sharing_dof)
                      {
                          case 2: {
                            // The DoF support point is in the middle of a
                            // primal edge at the boundary.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
                              dof_index_in_refined_mesh,
                              1.0);

                            break;
                          }
                          case 4: {
                            // The DoF support point is in the middle of a
                            // primal edge in the interior.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
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
                    dof_index_in_refined_mesh = dof_indices_in_cell[2];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();
                    this->averaging_matrix.set(dof_index_in_dual_mesh,
                                               dof_index_in_refined_mesh,
                                               1.0 /
                                                 number_of_cells_sharing_dof);

                    // Vertex 3 in the child cell, which is in the middle of a
                    // primal edge.
                    dof_index_in_refined_mesh = dof_indices_in_cell[3];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();

                    switch (number_of_cells_sharing_dof)
                      {
                          case 2: {
                            // The DoF support point is in the middle of a
                            // primal edge at the boundary.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
                              dof_index_in_refined_mesh,
                              1.0);

                            break;
                          }
                          case 4: {
                            // The DoF support point is in the middle of a
                            // primal edge in the interior.
                            this->averaging_matrix.set(
                              dof_index_in_dual_mesh,
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
                    dof_index_in_refined_mesh = dof_indices_in_cell[3];
                    number_of_cells_sharing_dof =
                      this->dof_to_cell_topo_dual_space
                        .topology[dof_index_in_refined_mesh]
                        .size();
                    this->averaging_matrix.set(dof_index_in_dual_mesh,
                                               dof_index_in_refined_mesh,
                                               1.0 /
                                                 number_of_cells_sharing_dof);

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
} // namespace HierBEM

#endif
