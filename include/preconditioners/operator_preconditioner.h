/**
 * @file operator_preconditioner.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef INCLUDE_OPERATOR_PRECONDITIONER_H_
#define INCLUDE_OPERATOR_PRECONDITIONER_H_

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <vector>

#include "bem_general.h"
#include "dof_tools_ext.h"

namespace HierBEM
{
  using namespace dealii;

  /**
   * Abstract class for operator preconditioning.
   *
   * \myref{Hiptmair, Ralf, and Carolina Urzua-Torres. 2016. “Dual Mesh Operator
   * Preconditioning On 3D Screens: Low-Order Boundary Element Discretization.”
   * 2016–14. CH-8092 Zürich, Switzerland: Seminar für Angewandte Mathematik,
   * Eidgenössische Technische Hochschule.}
   */
  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename MatrixType,
            typename RangeNumberType>
  class OperatorPreconditioner
  {
  public:
    /**
     * Constructor.
     */
    OperatorPreconditioner(FiniteElement<dim, spacedim>       &fe_primal_space,
                           FiniteElement<dim, spacedim>       &fe_dual_space,
                           const Triangulation<dim, spacedim> &primal_tria);

    ~OperatorPreconditioner();

    void
    setup_preconditioner();

    /**
     * Calculate matrix-vector multiplication as \f$y = C^{-1} \cdot x\f$, where
     * \f$C\f$ is the preconditioning matrix.
     *
     * @param y
     * @param x
     */
    void
    vmult(Vector<RangeNumberType> &y, const Vector<RangeNumberType> &x) const;

    /**
     * Build the coupling matrix \f$C_p\f$.
     */
    virtual void
    build_coupling_matrix() = 0;

    /**
     * Build the averaging matrix \f$C_d\f$.
     */
    virtual void
    build_averaging_matrix() = 0;

    /**
     * Build the mass matrix on the refined mesh.
     */
    virtual void
    build_mass_matrix_on_refined_mesh(const Quadrature<dim> &quad_rule);

    virtual void
    build_preconditioning_hmat_on_refined_mesh() = 0;

    void
    solve_mass_matrix(Vector<RangeNumberType>       &y,
                      const Vector<RangeNumberType> &x);

    void
    solve_mass_matrix_transpose(Vector<RangeNumberType>       &y,
                                const Vector<RangeNumberType> &x);

    SparseMatrix<RangeNumberType> &
    get_coupling_matrix()
    {
      return coupling_matrix;
    }

    const SparseMatrix<RangeNumberType> &
    get_coupling_matrix() const
    {
      return coupling_matrix;
    }

    SparseMatrix<RangeNumberType> &
    get_averaging_matrix()
    {
      return averaging_matrix;
    }

    const SparseMatrix<RangeNumberType> &
    get_averaging_matrix() const
    {
      return averaging_matrix;
    }

    SparseMatrix<RangeNumberType> &
    get_mass_matrix()
    {
      return mass_matrix;
    }

    const SparseMatrix<RangeNumberType> &
    get_mass_matrix() const
    {
      return mass_matrix;
    }

    Triangulation<dim, spacedim> &
    get_triangulation()
    {
      return tria;
    }

    const Triangulation<dim, spacedim> &
    get_triangulation() const
    {
      return tria;
    }

    DoFHandler<dim, spacedim> &
    get_dof_handler_primal_space()
    {
      return dof_handler_primal_space;
    }

    const DoFHandler<dim, spacedim> &
    get_dof_handler_primal_space() const
    {
      return dof_handler_primal_space;
    }

    DoFHandler<dim, spacedim> &
    get_dof_handler_dual_space()
    {
      return dof_handler_dual_space;
    }

    const DoFHandler<dim, spacedim> &
    get_dof_handler_dual_space() const
    {
      return dof_handler_dual_space;
    }

  protected:
    /**
     * Triangulation with two levels, i.e. orginal mesh and its refinement.
     */
    Triangulation<dim, spacedim> tria;
    /**
     * Coupling matrix \f$C_p\f$, which maps from the primal space on the
     * refined mesh \f$\bar{\Gamma}_h\f$ to the primal space on the primal mesh
     * \f$\Gamma_h\f$.
     *
     * The primal space is the domain space of the original operator to be
     * preconditioned. The dual space, as the name suggests, is the dual space
     * of this primal space. The dual space is also the domain space of the
     * preconditioning operator.
     */
    SparseMatrix<RangeNumberType> coupling_matrix;

    /**
     * Sparsity pattern for the @p coupling_matrix .
     */
    SparsityPattern coupling_matrix_sp;

    /**
     * Averaging matrix \f$C_d\f$, which maps from the dual space on the refined
     * mesh \f$\bar{\Gamma}_h\f$ to the dual space on the dual mesh
     * \f$\hat{\Gamma}_h\f$.
     */
    SparseMatrix<RangeNumberType> averaging_matrix;

    /**
     * Sparsity pattern for the @p averaging_matrix .
     */
    SparsityPattern averaging_matrix_sp;

    /**
     * Mass matrix associated with the preconditioning operator, which maps from
     * the range space of the preconditioner to its dual. This is equivalent to
     * say it maps from the primal space of the original operator to its dual
     * space.
     */
    SparseMatrix<RangeNumberType> mass_matrix;


    /**
     * Sparsity pattern for the @p mass_matrix .
     */
    SparsityPattern mass_matrix_sp;

    /**
     * Kernel function for the preconditioner.
     */
    KernelFunctionType preconditioner_kernel;
    /**
     * The Galerkin matrix for the preconditioner constructed on the refined
     * mesh.
     */
    MatrixType preconditioning_mat;
    /**
     * Finite element for the primal space.
     */
    FiniteElement<dim, spacedim> &fe_primal_space;
    /**
     * Finite element for the dual space.
     */
    FiniteElement<dim, spacedim> &fe_dual_space;
    /**
     * DoF handler for the finite element on the primal space.
     *
     * Because the triangulation has two levels, this DoF handler should be
     * distributed its DoFs to the multigrid.
     */
    DoFHandler<dim, spacedim> dof_handler_primal_space;
    /**
     * DoF handler for the finite element on the dual space.
     *
     * Because the triangulation has two levels, this DoF handler should be
     * distributed its DoFs to the multigrid.
     */
    DoFHandler<dim, spacedim> dof_handler_dual_space;
    /**
     * Collection of cell iterators held in the DoF handler for the primal
     * space on the refined mesh.
     */
    std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
      cell_iterators_primal_space;
    /**
     * Collection of cell iterators held in the DoF handler for the dual space
     * on the refined mesh.
     */
    std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
      cell_iterators_dual_space;
    /**
     * DoF-to-cell topology for the primal space on the refined mesh.
     */
    DoFToolsExt::DoFToCellTopology<dim, spacedim> dof_to_cell_topo_primal_space;
    /**
     * DoF-to-cell topology for the dual space on the refined mesh.
     */
    DoFToolsExt::DoFToCellTopology<dim, spacedim> dof_to_cell_topo_dual_space;
  };


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename MatrixType,
            typename RangeNumberType>
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         MatrixType,
                         RangeNumberType>::
    OperatorPreconditioner(FiniteElement<dim, spacedim>       &fe_primal_space,
                           FiniteElement<dim, spacedim>       &fe_dual_space,
                           const Triangulation<dim, spacedim> &primal_tria)
    : fe_primal_space(fe_primal_space)
    , fe_dual_space(fe_dual_space)
  {
    // Make a copy of the existing triangulation and perform a global
    // refinement.
    tria.copy_triangulation(primal_tria);
    tria.refine_global();

    // Initialize DoF handlers for primal and dual function spaces.
    dof_handler_primal_space.reinit(tria);
    dof_handler_dual_space.reinit(tria);

    dof_handler_primal_space.distribute_dofs(fe_primal_space);
    dof_handler_primal_space.distribute_mg_dofs();
    dof_handler_dual_space.distribute_dofs(fe_dual_space);
    dof_handler_dual_space.distribute_mg_dofs();

    // Collect cell iterators for the two function spaces on the refined mesh.
    cell_iterators_primal_space.reserve(dof_handler_primal_space.n_dofs(1));
    for (const auto &cell :
         dof_handler_primal_space.mg_cell_iterators_on_level(1))
      {
        cell_iterators_primal_space.push_back(cell);
      }

    cell_iterators_dual_space.reserve(dof_handler_dual_space.n_dofs(1));
    for (const auto &cell :
         dof_handler_dual_space.mg_cell_iterators_on_level(1))
      {
        cell_iterators_dual_space.push_back(cell);
      }

    // Generate DoF-to-cell topologies for the two function spaces on the
    // refined mesh.
    DoFToolsExt::build_mg_dof_to_cell_topology(dof_to_cell_topo_primal_space,
                                               cell_iterators_primal_space,
                                               dof_handler_primal_space,
                                               1);
    DoFToolsExt::build_mg_dof_to_cell_topology(dof_to_cell_topo_dual_space,
                                               cell_iterators_dual_space,
                                               dof_handler_dual_space,
                                               1);
  }


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename MatrixType,
            typename RangeNumberType>
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         MatrixType,
                         RangeNumberType>::~OperatorPreconditioner()
  {
    dof_handler_primal_space.clear();
    dof_handler_dual_space.clear();
  }


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename MatrixType,
            typename RangeNumberType>
  void
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         MatrixType,
                         RangeNumberType>::vmult(Vector<RangeNumberType> &y,
                                                 const Vector<RangeNumberType>
                                                   &x) const
  {}


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename MatrixType,
            typename RangeNumberType>
  void
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         MatrixType,
                         RangeNumberType>::
    build_mass_matrix_on_refined_mesh(const Quadrature<dim> &quad_rule)
  {
    // Generate the sparsity pattern for the mass matrix.
    DynamicSparsityPattern dsp(dof_handler_dual_space.n_dofs(1),
                               dof_handler_primal_space.n_dofs(1));
    DoFTools::make_sparsity_pattern(dof_handler_dual_space,
                                    dof_handler_primal_space,
                                    dsp);
    mass_matrix_sp.copy_from(dsp);
    mass_matrix.reinit(mass_matrix_sp);

    // Assemble the mass matrix.
    assemble_fem_scaled_mass_matrix(dof_handler_dual_space,
                                    dof_handler_primal_space,
                                    1.0,
                                    quad_rule,
                                    mass_matrix);
  }

  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename MatrixType,
            typename RangeNumberType>
  void
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         MatrixType,
                         RangeNumberType>::setup_preconditioner()
  {
    build_preconditioning_hmat_on_refined_mesh();
    build_mass_matrix_on_refined_mesh();

    build_coupling_matrix();
    build_averaging_matrix();
  }
} // namespace HierBEM

#endif /* INCLUDE_OPERATOR_PRECONDITIONER_H_ */