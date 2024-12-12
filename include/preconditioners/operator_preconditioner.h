/**
 * @file operator_preconditioner.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef INCLUDE_OPERATOR_PRECONDITIONER_H_
#define INCLUDE_OPERATOR_PRECONDITIONER_H_

#include <deal.II/base/parallel.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operations_internal.h>

#include <fstream>
#include <map>
#include <vector>

#include "aca_plus.hcu"
#include "bem_general.h"
#include "bem_kernels.hcu"
#include "block_cluster_tree.h"
#include "dof_tools_ext.h"
#include "generic_functors.h"
#include "grid_out_ext.h"
#include "hmatrix.h"
#include "hmatrix_symm.h"
#include "mapping/mapping_info.h"
#include "sauter_quadrature_tools.h"
#include "subdomain_topology.h"

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
            typename RangeNumberType,
            typename SurfaceNormalDetector>
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
    setup_preconditioner(
      const unsigned int                               thread_num,
      const HMatrixParameters                         &hmat_params,
      const std::vector<MappingInfo<dim, spacedim> *> &mappings,
      const std::map<types::material_id, unsigned int>
                                      &material_id_to_mapping_index,
      const SurfaceNormalDetector     &normal_detector,
      const SauterQuadratureRule<dim> &sauter_quad_rule,
      const Quadrature<dim>           &quad_rule_for_mass);

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

    /**
     * @brief Build the preconditioning \hmat on the refined mesh.
     *
     * \alert{This function should be called after @p build_mass_matrix_on_refined_mesh ,
     * since it relies on the mass matrix when building the preconditioning
     * \hmat for the single layer potential operator.}
     *
     * @param thread_num Number of CPU threads
     * @param hmat_params \hmat parameters
     * @param mappings A list of pointers for MappingInfo objects of different
     * orders.
     * @param material_id_to_mapping_index Map from @p material_id to index for accessing @p mappings.
     * @param normal_detector Object for detecting the surface normal direction
     * of a cell.
     * @param sauter_quad_rule Sauter quadrature rule
     */
    virtual void
    build_preconditioning_hmat_on_refined_mesh(
      const unsigned int                               thread_num,
      const HMatrixParameters                         &hmat_params,
      const std::vector<MappingInfo<dim, spacedim> *> &mappings,
      const std::map<types::material_id, unsigned int>
                                      &material_id_to_mapping_index,
      const SurfaceNormalDetector     &normal_detector,
      const SauterQuadratureRule<dim> &sauter_quad_rule);

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

    HMatrixSymm<spacedim, RangeNumberType> &
    get_preconditioning_hmatrix()
    {
      return preconditioning_hmat;
    }

    const HMatrixSymm<spacedim, RangeNumberType> &
    get_preconditioning_hmatrix() const
    {
      return preconditioning_hmat;
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
     * the range space of the preconditioner to its dual space. This is
     * equivalent to say that it maps from the primal space of the original
     * operator on the refined mesh \f$\bar{\Gamma}_h\f$ to the dual space on
     * the refined mesh.
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
     * The Galerkin matrix for the preconditioner. This matrix maps from the
     * dual space on the refined mesh \f$\bar{\Gamma}_h\f$ to itself.
     */
    HMatrixSymm<spacedim, RangeNumberType> preconditioning_hmat;

    /**
     * Cluster tree for the preconditioning \hmat.
     */
    ClusterTree<spacedim> ct;

    /**
     * Block cluster tree for the preconditioning \hmat.
     */
    BlockClusterTree<spacedim> bct;

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
            typename RangeNumberType,
            typename SurfaceNormalDetector>
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         RangeNumberType,
                         SurfaceNormalDetector>::
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
            typename RangeNumberType,
            typename SurfaceNormalDetector>
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         RangeNumberType,
                         SurfaceNormalDetector>::~OperatorPreconditioner()
  {
    dof_handler_primal_space.clear();
    dof_handler_dual_space.clear();
  }


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename RangeNumberType,
            typename SurfaceNormalDetector>
  void
  OperatorPreconditioner<
    dim,
    spacedim,
    KernelFunctionType,
    RangeNumberType,
    SurfaceNormalDetector>::vmult(Vector<RangeNumberType>       &y,
                                  const Vector<RangeNumberType> &x) const
  {}


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename RangeNumberType,
            typename SurfaceNormalDetector>
  void
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         RangeNumberType,
                         SurfaceNormalDetector>::
    build_mass_matrix_on_refined_mesh(const Quadrature<dim> &quad_rule)
  {
    // Generate the sparsity pattern for the mass matrix.
    DynamicSparsityPattern dsp(dof_handler_dual_space.n_dofs(1),
                               dof_handler_primal_space.n_dofs(1));
    // N.B. DoFTools::make_sparsity_pattern operates on active cells, which just
    // corresponds to the refined mesh that we desire.
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
            typename RangeNumberType,
            typename SurfaceNormalDetector>
  void
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         RangeNumberType,
                         SurfaceNormalDetector>::
    build_preconditioning_hmat_on_refined_mesh(
      const unsigned int                               thread_num,
      const HMatrixParameters                         &hmat_params,
      const std::vector<MappingInfo<dim, spacedim> *> &mappings,
      const std::map<types::material_id, unsigned int>
                                      &material_id_to_mapping_index,
      const SurfaceNormalDetector     &normal_detector,
      const SauterQuadratureRule<dim> &sauter_quad_rule)
  {
    /**
     * Generate lists of DoF indices.
     */
    const unsigned int n_dofs = dof_handler_dual_space.n_dofs(1);
    std::vector<types::global_dof_index> dof_indices_in_dual_space(n_dofs);
    gen_linear_indices<vector_uta, types::global_dof_index>(
      dof_indices_in_dual_space);

    /**
     * Get the spatial coordinates of the support points.
     */
    std::vector<Point<spacedim>> support_points_in_dual_space(n_dofs);
    DoFToolsExt::map_mg_dofs_to_support_points(mappings[0]->get_mapping(),
                                               dof_handler_dual_space,
                                               1,
                                               support_points_in_dual_space);

    /**
     * Calculate the average mesh cell size at each support point.
     */
    std::vector<double> dof_average_cell_size_list(n_dofs, 0.0);
    DoFToolsExt::map_mg_dofs_to_average_cell_size(dof_handler_dual_space,
                                                  1,
                                                  dof_average_cell_size_list);

    /**
     * Initialize the cluster tree.
     */
    ct = ClusterTree<spacedim>(dof_indices_in_dual_space,
                               support_points_in_dual_space,
                               dof_average_cell_size_list,
                               hmat_params.n_min_for_ct);

    /**
     * Partition the cluster tree.
     */
    ct.partition(support_points_in_dual_space, dof_average_cell_size_list);

    /**
     * Get the internal-to-external DoF numberings.
     */
    std::vector<types::global_dof_index> &dof_i2e_numbering =
      ct.get_internal_to_external_dof_numbering();

    /**
     * Create the block cluster tree.
     */
    bct = BlockClusterTree<spacedim>(ct,
                                     ct,
                                     hmat_params.eta,
                                     hmat_params.n_min_for_bct);

    /**
     * Partition the block cluster tree.
     */
    bct.partition(dof_i2e_numbering,
                  support_points_in_dual_space,
                  dof_average_cell_size_list);

    preconditioning_hmat =
      HMatrixSymm<spacedim, RangeNumberType>(bct, hmat_params.max_hmat_rank);

    ACAConfig aca_config(hmat_params.max_hmat_rank,
                         hmat_params.aca_relative_error,
                         hmat_params.eta);

    /**
     * When the kernel function is the hyper-singular function, which has a
     * non-trival kernel space \f$\mathrm{span}\{1\}\f$, we need to add a
     * regularization term to the bilinear form. In other cases, regularization
     * is not needed.
     */
    if (preconditioner_kernel.kernel_type == KernelType::HyperSingularRegular)
      {
        /**
         * Set natural density as constant vector 1 and set the alpha factor
         * as 1.
         */
        Vector<RangeNumberType> natural_density(
          dof_handler_primal_space.n_dofs(1));
        dealii::internal::VectorOperations::Vector_set<double> setter(
          1.0, natural_density.begin());
        auto partitioner =
          std::make_shared<dealii::parallel::internal::TBBPartitioner>();
        dealii::internal::VectorOperations::parallel_for(setter,
                                                         0,
                                                         natural_density.size(),
                                                         partitioner);
        const double alpha_for_neumann = 1.0;

        /**
         * Calculate the vector \f$a\f$ in \f$\alpha a a^T\f$, where \f$a\f$
         * is the multiplication of the mass matrix and the natural density.
         *
         * N.B. The mass matrix on the refined mesh maps from the primal
         * function space to the dual space, which is consistent with the vector
         * size of the input vector @p natural_density and the output vector
         * @p mass_vmult_weq.
         */
        Vector<RangeNumberType> mass_vmult_weq(n_dofs);
        mass_matrix.vmult(mass_vmult_weq, natural_density);

        /**
         * Assemble the preconditioning matrix.
         */
        fill_hmatrix_with_aca_plus_smp(thread_num,
                                       preconditioning_hmat,
                                       aca_config,
                                       preconditioner_kernel,
                                       1.0,
                                       mass_vmult_weq,
                                       alpha_for_neumann,
                                       dof_to_cell_topo_dual_space,
                                       dof_to_cell_topo_dual_space,
                                       sauter_quad_rule,
                                       dof_handler_dual_space,
                                       dof_handler_dual_space,
                                       nullptr,
                                       nullptr,
                                       dof_i2e_numbering,
                                       dof_i2e_numbering,
                                       mappings,
                                       material_id_to_mapping_index,
                                       normal_detector,
                                       true);
      }
    else
      {
        fill_hmatrix_with_aca_plus_smp(thread_num,
                                       preconditioning_hmat,
                                       aca_config,
                                       preconditioner_kernel,
                                       1.0,
                                       dof_to_cell_topo_dual_space,
                                       dof_to_cell_topo_dual_space,
                                       sauter_quad_rule,
                                       dof_handler_dual_space,
                                       dof_handler_dual_space,
                                       nullptr,
                                       nullptr,
                                       dof_i2e_numbering,
                                       dof_i2e_numbering,
                                       mappings,
                                       material_id_to_mapping_index,
                                       normal_detector,
                                       true);
      }
  }


  template <int dim,
            int spacedim,
            typename KernelFunctionType,
            typename RangeNumberType,
            typename SurfaceNormalDetector>
  void
  OperatorPreconditioner<dim,
                         spacedim,
                         KernelFunctionType,
                         RangeNumberType,
                         SurfaceNormalDetector>::
    setup_preconditioner(
      const unsigned int                               thread_num,
      const HMatrixParameters                         &hmat_params,
      const std::vector<MappingInfo<dim, spacedim> *> &mappings,
      const std::map<types::material_id, unsigned int>
                                      &material_id_to_mapping_index,
      const SurfaceNormalDetector     &normal_detector,
      const SauterQuadratureRule<dim> &sauter_quad_rule,
      const Quadrature<dim>           &quad_rule_for_mass)
  {
    build_coupling_matrix();
    build_averaging_matrix();
    build_mass_matrix_on_refined_mesh(quad_rule_for_mass);
    build_preconditioning_hmat_on_refined_mesh(thread_num,
                                               hmat_params,
                                               mappings,
                                               material_id_to_mapping_index,
                                               normal_detector,
                                               sauter_quad_rule);
  }
} // namespace HierBEM

#endif /* INCLUDE_OPERATOR_PRECONDITIONER_H_ */