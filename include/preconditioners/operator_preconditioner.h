/**
 * @file operator_preconditioner.h
 * @brief
 *
 * @date 2024-12-02
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_PRECONDITIONERS_OPERATOR_PRECONDITIONER_H_
#define HIERBEM_INCLUDE_PRECONDITIONERS_OPERATOR_PRECONDITIONER_H_

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operations_internal.h>

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "bem_general.hcu"
#include "bem_kernels.hcu"
#include "block_cluster_tree.h"
#include "config.h"
#include "debug_tools.h"
#include "dof_to_cell_topology.h"
#include "dof_tools_ext.h"
#include "generic_functors.h"
#include "grid_out_ext.h"
#include "hmatrix/aca_plus/aca_plus.hcu"
#include "hmatrix/hmatrix.h"
#include "hmatrix/hmatrix_parameters.h"
#include "mapping/mapping_info.h"
#include "sauter_quadrature_tools.h"
#include "subdomain_topology.h"

HBEM_NS_OPEN

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
          typename RangeNumberType>
class OperatorPreconditioner
{
public:
  /**
   * Constructor.
   */
  OperatorPreconditioner(
    const std::string                          &type_,
    FiniteElement<dim, spacedim>               &fe_primal_space_,
    FiniteElement<dim, spacedim>               &fe_dual_space_,
    const Triangulation<dim, spacedim>         &tria_,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering_,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering_,
    const std::set<types::material_id>         &subdomain_material_ids_ =
      std::set<types::material_id>(),
    const std::set<types::material_id> &subdomain_complement_material_ids_ =
      std::set<types::material_id>(),
    const bool is_full_domain_                                       = true,
    const bool is_subdomain_open_                                    = false,
    const bool truncate_function_space_dof_support_within_subdomain_ = false,
    const unsigned int max_iter                                      = 1000,
    const double       tol                                           = 1e-8,
    const double       omega                                         = 1.0,
    const bool         log_history                                   = true,
    const bool         log_result                                    = true);

  virtual ~OperatorPreconditioner();

  /**
   * Initialize primal and dual space DoF handlers on the multigrid.
   */
  void
  initialize_dof_handlers();

  /**
   * Generate DoF selectors. This function is used when the preconditioner is
   * defined on a subdomain.
   */
  void
  generate_dof_selectors();

  /**
   * Generate maps between full and local DoF indices for primal and dual
   * spaces, on either refined or primal mesh. This function is used when the
   * preconditioner is defined on a subdomain.
   */
  void
  generate_maps_between_full_and_local_dof_ids();

  /**
   * Build DoF-to-cell topology for the dual space on the refined mesh.
   *
   * DoF-to-cell topology will be used for building the \hmat inolving the dual
   * space on the refined mesh.
   *
   * \alert{2025-02-07 According to \myref{Hiptmair, Ralf, and Carolina
   * Urzua-Torres. 2016. “Dual Mesh Operator Preconditioning On 3D Screens:
   * Low-Order Boundary Element Discretization.” 2016–14. CH-8092 Zürich,
   * Switzerland: Seminar für Angewandte Mathematik, Eidgenössische Technische
   * Hochschule.}, the dual space is \f$H^{1/2}(\Gamma_{\mathrm{D}})\f$ in
   * Dirichlet problem, which means the finite element basis functions do not
   * penetrate into the adjacent subdomain \f$\Gamma_{\mathrm{N}}\f$. If we
   * enforce this condition by restricting the related DoF-to-cell topology
   * within \f$\Gamma_{\mathrm{D}}\f$, the preconditioned matrix will be
   * ill-posed. Therefore, we remove this restriction and construct the
   * DoF-to-cell topology by iterating over all cells in the triangulation.}
   */
  virtual void
  build_dof_to_cell_topology();

  template <typename SurfaceNormalDetector>
  void
  setup_preconditioner(
    const unsigned int                               thread_num,
    const HMatrixParameters                         &hmat_params,
    const SubdomainTopology<dim, spacedim>          &subdomain_topology,
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
   * Reinitialize intermediate vectors.
   *
   * This function should be called after building the matrices used in
   * operator preconditioning.
   */
  void
  reinit();

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
  void
  build_mass_matrix_on_refined_mesh(const Quadrature<dim> &quad_rule);

  /**
   * @brief Build the cluster and block cluster trees for the \hmat to be
   * constructed involving the dual space on the refined mesh.
   */
  void
  build_cluster_and_block_cluster_trees(
    const HMatrixParameters                         &hmat_params,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings);

  /**
   * @brief Build the Galerkin \hmat for the preconditioner on the refined
   * mesh.
   *
   * \alert{This function should be called after @p build_mass_matrix_on_refined_mesh ,
   * since it relies on the mass matrix when building the preconditioning
   * \hmat for the single layer potential operator.}
   *
   * @param thread_num Number of CPU threads
   * @param hmat_params \hmat parameters
   * @param mappings A list of pointers for MappingInfo objects of different
   * orders.
   * @param material_id_to_mapping_index Map from @p material_id to index for accessing
   * @p mappings.
   * @param normal_detector Object for detecting the surface normal direction
   * of a cell.
   * @param sauter_quad_rule Sauter quadrature rule
   */
  template <typename SurfaceNormalDetector>
  void
  build_preconditioner_hmat_on_refined_mesh(
    const unsigned int                               thread_num,
    const HMatrixParameters                         &hmat_params,
    const SubdomainTopology<dim, spacedim>          &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule);

  void
  solve_mass_matrix_triple(Vector<RangeNumberType>       &y,
                           const Vector<RangeNumberType> &x) const;

  void
  solve_mass_matrix_transpose_triple(Vector<RangeNumberType>       &y,
                                     const Vector<RangeNumberType> &x) const;

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

  HMatrix<spacedim, RangeNumberType> &
  get_preconditioner_hmatrix()
  {
    return preconditioner_hmat;
  }

  const HMatrix<spacedim, RangeNumberType> &
  get_preconditioner_hmatrix() const
  {
    return preconditioner_hmat;
  }

  Vector<RangeNumberType> &
  get_mass_matrix_triple_diag_reciprocal()
  {
    return mass_matrix_triple_diag_reciprocal;
  }

  const Vector<RangeNumberType> &
  get_mass_matrix_triple_diag_reciprocal() const
  {
    return mass_matrix_triple_diag_reciprocal;
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

  std::set<types::material_id> &
  get_subdomain_material_ids()
  {
    return subdomain_material_ids;
  }

  const std::set<types::material_id> &
  get_subdomain_material_ids() const
  {
    return subdomain_material_ids;
  }

  std::vector<bool> &
  get_primal_space_dof_selectors_on_primal_mesh()
  {
    return primal_space_dof_selectors_on_primal_mesh;
  }

  const std::vector<bool> &
  get_primal_space_dof_selectors_on_primal_mesh() const
  {
    return primal_space_dof_selectors_on_primal_mesh;
  }

  std::vector<bool> &
  get_primal_space_dof_selectors_on_refined_mesh()
  {
    return primal_space_dof_selectors_on_refined_mesh;
  }

  const std::vector<bool> &
  get_primal_space_dof_selectors_on_refined_mesh() const
  {
    return primal_space_dof_selectors_on_refined_mesh;
  }

  std::vector<bool> &
  get_dual_space_dof_selectors_on_refined_mesh()
  {
    return dual_space_dof_selectors_on_refined_mesh;
  }

  const std::vector<bool> &
  get_dual_space_dof_selectors_on_refined_mesh() const
  {
    return dual_space_dof_selectors_on_refined_mesh;
  }

  std::vector<types::global_dof_index> &
  get_primal_space_full_to_local_dof_id_map_on_primal_mesh()
  {
    return primal_space_full_to_local_dof_id_map_on_primal_mesh;
  }

  const std::vector<types::global_dof_index> &
  get_primal_space_full_to_local_dof_id_map_on_primal_mesh() const
  {
    return primal_space_full_to_local_dof_id_map_on_primal_mesh;
  }

  std::vector<types::global_dof_index> &
  get_primal_space_full_to_local_dof_id_map_on_refined_mesh()
  {
    return primal_space_full_to_local_dof_id_map_on_refined_mesh;
  }

  const std::vector<types::global_dof_index> &
  get_primal_space_full_to_local_dof_id_map_on_refined_mesh() const
  {
    return primal_space_full_to_local_dof_id_map_on_refined_mesh;
  }

  std::vector<types::global_dof_index> &
  get_dual_space_full_to_local_dof_id_map_on_refined_mesh()
  {
    return dual_space_full_to_local_dof_id_map_on_refined_mesh;
  }

  const std::vector<types::global_dof_index> &
  get_dual_space_full_to_local_dof_id_map_on_refined_mesh() const
  {
    return dual_space_full_to_local_dof_id_map_on_refined_mesh;
  }

protected:
  /**
   * @brief This class encapsulates the product of three matrices: \f$C_d M_r
   * C_p^{\mathrm{T}}\f$.
   */
  class MassMatrixTriple : public Subscriptor
  {
  public:
    MassMatrixTriple(
      SparseMatrix<RangeNumberType> &coupling_matrix,
      SparseMatrix<RangeNumberType> &averaging_matrix,
      SparseMatrix<RangeNumberType> &mass_matrix,
      Vector<RangeNumberType>       &mass_matrix_triple_diag_reciprocal)
      : coupling_matrix(coupling_matrix)
      , averaging_matrix(averaging_matrix)
      , mass_matrix(mass_matrix)
      , mass_matrix_triple_diag_reciprocal(mass_matrix_triple_diag_reciprocal)
    {
      v1 = new Vector<RangeNumberType>();
      v2 = new Vector<RangeNumberType>();
    }

    ~MassMatrixTriple()
    {
      delete v1;
      delete v2;
    }

    /**
     * Reinitialize intermediate vectors.
     *
     * This function should be called after building the matrices used in
     * operator preconditioning.
     */
    void
    reinit()
    {
      v1->reinit(coupling_matrix.n());
      v2->reinit(mass_matrix.m());
    }

    void
    vmult(Vector<RangeNumberType> &y, const Vector<RangeNumberType> &x) const
    {
      AssertDimension(coupling_matrix.m(), x.size());
      AssertDimension(averaging_matrix.m(), y.size());
      AssertDimension(x.size(), y.size());

      coupling_matrix.Tvmult(*v1, x);
      mass_matrix.vmult(*v2, *v1);
      averaging_matrix.vmult(y, *v2);
    }

    void
    precondition_Jacobi(Vector<RangeNumberType>       &dst,
                        const Vector<RangeNumberType> &src,
                        const RangeNumberType          omega = 1.) const
    {
      using size_type = typename Vector<RangeNumberType>::size_type;

      AssertDimension(dst.size(), src.size());

      const size_type        n       = src.size();
      RangeNumberType       *dst_ptr = dst.begin();
      const RangeNumberType *src_ptr = src.begin();
      const RangeNumberType *diag_ptr =
        mass_matrix_triple_diag_reciprocal.begin();

      if (omega == 1.0)
        for (size_type i = 0; i < n; i++, dst_ptr++, src_ptr++, diag_ptr++)
          *dst_ptr = *src_ptr * (*diag_ptr);
      else
        for (size_type i = 0; i < n; i++, dst_ptr++, src_ptr++, diag_ptr++)
          *dst_ptr = *src_ptr * (*diag_ptr) * omega;
    }

  private:
    SparseMatrix<RangeNumberType> &coupling_matrix;
    SparseMatrix<RangeNumberType> &averaging_matrix;
    SparseMatrix<RangeNumberType> &mass_matrix;
    Vector<RangeNumberType>       &mass_matrix_triple_diag_reciprocal;

    /**
     * The result vector of \f$C_p^{\mathrm{T}}x\f$.
     */
    Vector<RangeNumberType> *v1;
    /**
     * The result vector of \f$M_r v_1\f$.
     */
    Vector<RangeNumberType> *v2;
  };

  /**
   * @brief This class encapsulates the transpose of the product of three
   * matrices: \f$C_d M_r C_p^{\mathrm{T}}\f$, which is actually \f$C_p
   * M_r^{\mathrm{T}} C_d^{\mathrm{T}}\f$.
   */
  class MassMatrixTransposeTriple : public Subscriptor
  {
  public:
    MassMatrixTransposeTriple(
      SparseMatrix<RangeNumberType> &coupling_matrix,
      SparseMatrix<RangeNumberType> &averaging_matrix,
      SparseMatrix<RangeNumberType> &mass_matrix,
      Vector<RangeNumberType>       &mass_matrix_triple_diag_reciprocal)
      : coupling_matrix(coupling_matrix)
      , averaging_matrix(averaging_matrix)
      , mass_matrix(mass_matrix)
      , mass_matrix_triple_diag_reciprocal(mass_matrix_triple_diag_reciprocal)
    {
      v1 = new Vector<RangeNumberType>();
      v2 = new Vector<RangeNumberType>();
    }

    ~MassMatrixTransposeTriple()
    {
      delete v1;
      delete v2;
    }

    /**
     * Reinitialize intermediate vectors.
     *
     * This function should be called after building the matrices used in
     * operator preconditioning.
     */
    void
    reinit()
    {
      v1->reinit(averaging_matrix.n());
      v2->reinit(mass_matrix.n());
    }

    void
    vmult(Vector<RangeNumberType> &y, const Vector<RangeNumberType> &x) const
    {
      AssertDimension(averaging_matrix.m(), x.size());
      AssertDimension(coupling_matrix.m(), y.size());
      AssertDimension(x.size(), y.size());

      averaging_matrix.Tvmult(*v1, x);
      mass_matrix.Tvmult(*v2, *v1);
      coupling_matrix.vmult(y, *v2);
    }

    void
    precondition_Jacobi(Vector<RangeNumberType>       &dst,
                        const Vector<RangeNumberType> &src,
                        const RangeNumberType          omega = 1.) const
    {
      using size_type = typename Vector<RangeNumberType>::size_type;

      AssertDimension(dst.size(), src.size());

      const size_type        n       = src.size();
      RangeNumberType       *dst_ptr = dst.begin();
      const RangeNumberType *src_ptr = src.begin();
      const RangeNumberType *diag_ptr =
        mass_matrix_triple_diag_reciprocal.begin();

      if (omega == 1.0)
        for (size_type i = 0; i < n; i++, dst_ptr++, src_ptr++, diag_ptr++)
          *dst_ptr = *src_ptr * (*diag_ptr);
      else
        for (size_type i = 0; i < n; i++, dst_ptr++, src_ptr++, diag_ptr++)
          *dst_ptr = *src_ptr * (*diag_ptr) * omega;
    }

  private:
    SparseMatrix<RangeNumberType> &coupling_matrix;
    SparseMatrix<RangeNumberType> &averaging_matrix;
    SparseMatrix<RangeNumberType> &mass_matrix;
    Vector<RangeNumberType>       &mass_matrix_triple_diag_reciprocal;

    /**
     * The result vector of \f$C_d^{\mathrm{T}}x\f$.
     */
    Vector<RangeNumberType> *v1;
    /**
     * The result vector of \f$M_r^{\mathrm{T}}v_1\f$.
     */
    Vector<RangeNumberType> *v2;
  };

  void
  compute_mass_matrix_triple_diag_reciprocal();

  /**
   * Preconditioner type string.
   */
  std::string type;

  /**
   * Triangulation with two levels, i.e. orginal mesh and its refinement.
   */
  const Triangulation<dim, spacedim> &tria;

  /**
   * If the preconditioner is built on the full domain.
   */
  bool is_full_domain;

  /**
   * When the preconditioner is not built on the full domain but a part of it,
   * this flag indicates if the subdomain is an open set. If so, the DoFs at the
   * interface (of course the finite element should have support points there)
   * between \f$\Gamma_{\mathrm{D}}\f$ and \f$\Gamma_{\mathrm{N}}\f$ are not
   * included.
   *
   * If the finite element is \f$L_2\f$ conforming, @p is_subdomain_open
   * being true or false has no influence.
   *
   * This variable is effective when @p is_full_domain is false.
   */
  bool is_subdomain_open;

  /**
   * This flag controls whether we truncate the support of basis functions
   * within the subdomain by constructing the DoF-to-cell topology without
   * those cells not in the current subdomain.
   */
  bool truncate_function_space_dof_support_within_subdomain;

  // Maps between full and local DoF indices, when the preconditioner is on a
  // subdomain. Because this is not a bijective map, in the current
  // implementation, the number of elements in the vector that maps from the
  // full to local DoF indices is equal to the total number of DoFs in the DoF
  // handler, while the number of elements in the vector that maps from the
  // local to full DoF indies is just the actual number of DoFs selected in the
  // subdomain.
  std::vector<types::global_dof_index>
    primal_space_full_to_local_dof_id_map_on_primal_mesh;
  std::vector<types::global_dof_index>
    primal_space_local_to_full_dof_id_map_on_primal_mesh;
  std::vector<types::global_dof_index>
    primal_space_full_to_local_dof_id_map_on_refined_mesh;
  std::vector<types::global_dof_index>
    primal_space_local_to_full_dof_id_map_on_refined_mesh;
  std::vector<types::global_dof_index>
    dual_space_full_to_local_dof_id_map_on_refined_mesh;
  std::vector<types::global_dof_index>
    dual_space_local_to_full_dof_id_map_on_refined_mesh;

  /**
   * A set of material ids in the subdomain. When @p is_full_domain is true, this
   * set is empty.
   */
  std::set<types::material_id> subdomain_material_ids;

  /**
   * A set of material ids in the complement of the subdomain. When
   * @p is_full_domain is true, this set is empty.
   */
  std::set<types::material_id> subdomain_complement_material_ids;

  // DoF selectors
  std::vector<bool> primal_space_dof_selectors_on_primal_mesh;
  std::vector<bool> primal_space_dof_selectors_on_refined_mesh;
  std::vector<bool> dual_space_dof_selectors_on_refined_mesh;

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
  HMatrix<spacedim, RangeNumberType> preconditioner_hmat;

  /**
   * Cluster tree for the preconditioning \hmat.
   */
  ClusterTree<spacedim> ct;

  /**
   * Block cluster tree for the preconditioning \hmat.
   */
  BlockClusterTree<spacedim> bct;

  /**
   * The internal-to-external DoF numbering for the primal space on the primal
   * mesh.
   */
  const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering;
  /**
   * The external-to-internal DoF numbering for the primal space on the primal
   * mesh.
   */
  const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering;

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
   * Collection of cell iterators held in the DoF handler for the dual space
   * on the refined mesh.
   */
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
    cell_iterators_dual_space;
  /**
   * DoF-to-cell topology for the dual space on the refined mesh.
   */
  DoFToCellTopology<dim, spacedim> dof_to_cell_topo_dual_space;

  // Intermediate vectors during @p vmult. The input vector is @p x and the
  // final output vector is @p y. Both @p x and @p y adopt the internal DoF
  // numbering.
  /**
   * The input vector @p x of @p vmult in the external DoF numbering.
   */
  Vector<RangeNumberType> *x_external_dof_numbering;
  /**
   * The result vector of \f$\(C_d M_r C_p^{\mathrm{T}}\)^{-\mathrm{T}}x\f$,
   * which is equivalent to solve \f$\(C_d M_r C_p^{\mathrm{T}}\)y=x\f$ for
   * \f$y\f$. This vector adopts the external DoF numbering.
   *
   * It is also used to store the result vector of \f$C_d v_3\f$.
   */
  Vector<RangeNumberType> *v1;
  /**
   * The result vector of \f$C_d^{\mathrm{T}} v_1\f$. It adtops the external
   * DoF numbering.
   */
  Vector<RangeNumberType> *v2;
  /**
   * The result vector of \f$C_d^{\mathrm{T}} v_1\f$ in the internal DoF
   * numbering, which is to be multiplied by \f$B_r\f$.
   */
  Vector<RangeNumberType> *v2_internal_dof_numbering;
  /**
   * @brief The result vector of \f$B_r v_2\f$, where \f$B_r\f$ is the \hmat
   * of the preconditioner on the refined mesh. It adopts the internal DoF
   * numbering.
   */
  Vector<RangeNumberType> *v3;
  /**
   * The result vector of \f$B_r v_2\f$ in the external DoF numbering, which
   * is to be multiplied by \f$C_d\f$.
   */
  Vector<RangeNumberType> *v3_external_dof_numbering;
  /**
   * The result vector of the whole @p vmult in the external DoF numbering,
   * which should be further converted to the internal DoF numbering.
   */
  Vector<RangeNumberType> *y_external_dof_numbering;

  // Diagonal entries in @p mass_matrix_triple.
  Vector<RangeNumberType> mass_matrix_triple_diag_reciprocal;

  MassMatrixTriple          mass_matrix_triple;
  MassMatrixTransposeTriple mass_matrix_transpose_triple;
  unsigned int              solve_mass_matrix_max_iter;
  double                    solve_mass_matrix_tol;
  double                    solve_mass_matrix_relaxation;
  bool                      solve_mass_matrix_log_history;
  bool                      solve_mass_matrix_log_result;
};


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  OperatorPreconditioner(
    const std::string                          &type_,
    FiniteElement<dim, spacedim>               &fe_primal_space_,
    FiniteElement<dim, spacedim>               &fe_dual_space_,
    const Triangulation<dim, spacedim>         &tria_,
    const std::vector<types::global_dof_index> &primal_space_dof_i2e_numbering_,
    const std::vector<types::global_dof_index> &primal_space_dof_e2i_numbering_,
    const std::set<types::material_id>         &subdomain_material_ids_,
    const std::set<types::material_id> &subdomain_complement_material_ids_,
    const bool                          is_full_domain_,
    const bool                          is_subdomain_open_,
    const bool         truncate_function_space_dof_support_within_subdomain_,
    const unsigned int max_iter,
    const double       tol,
    const double       omega,
    const bool         log_history,
    const bool         log_result)
  : type(type_)
  , tria(tria_)
  , is_full_domain(is_full_domain_)
  , is_subdomain_open(is_subdomain_open_)
  , truncate_function_space_dof_support_within_subdomain(
      truncate_function_space_dof_support_within_subdomain_)
  , subdomain_material_ids(subdomain_material_ids_)
  , subdomain_complement_material_ids(subdomain_complement_material_ids_)
  , primal_space_dof_i2e_numbering(primal_space_dof_i2e_numbering_)
  , primal_space_dof_e2i_numbering(primal_space_dof_e2i_numbering_)
  , fe_primal_space(fe_primal_space_)
  , fe_dual_space(fe_dual_space_)
  , mass_matrix_triple(coupling_matrix,
                       averaging_matrix,
                       mass_matrix,
                       mass_matrix_triple_diag_reciprocal)
  , mass_matrix_transpose_triple(coupling_matrix,
                                 averaging_matrix,
                                 mass_matrix,
                                 mass_matrix_triple_diag_reciprocal)
  , solve_mass_matrix_max_iter(max_iter)
  , solve_mass_matrix_tol(tol)
  , solve_mass_matrix_relaxation(omega)
  , solve_mass_matrix_log_history(log_history)
  , solve_mass_matrix_log_result(log_result)
{
  x_external_dof_numbering  = new Vector<RangeNumberType>();
  v1                        = new Vector<RangeNumberType>();
  v2                        = new Vector<RangeNumberType>();
  v2_internal_dof_numbering = new Vector<RangeNumberType>();
  v3                        = new Vector<RangeNumberType>();
  v3_external_dof_numbering = new Vector<RangeNumberType>();
  y_external_dof_numbering  = new Vector<RangeNumberType>();
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  ~OperatorPreconditioner()
{
  dof_handler_primal_space.clear();
  dof_handler_dual_space.clear();

  delete x_external_dof_numbering;
  delete v1;
  delete v2;
  delete v2_internal_dof_numbering;
  delete v3;
  delete v3_external_dof_numbering;
  delete y_external_dof_numbering;
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  build_mass_matrix_on_refined_mesh(const Quadrature<dim> &quad_rule)
{
  // Generate the sparsity pattern for the mass matrix.
  if (is_full_domain)
    {
      DynamicSparsityPattern dsp(dof_handler_dual_space.n_dofs(1),
                                 dof_handler_primal_space.n_dofs(1));
      // N.B. DoFTools::make_sparsity_pattern operates on active cells, which
      // just corresponds to the refined mesh that we desire.
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
  else
    {
      DynamicSparsityPattern dsp(
        dual_space_local_to_full_dof_id_map_on_refined_mesh.size(),
        primal_space_local_to_full_dof_id_map_on_refined_mesh.size());
      DoFToolsExt::make_sparsity_pattern(
        dof_handler_dual_space,
        dof_handler_primal_space,
        subdomain_material_ids,
        dual_space_dof_selectors_on_refined_mesh,
        dual_space_full_to_local_dof_id_map_on_refined_mesh,
        primal_space_dof_selectors_on_refined_mesh,
        primal_space_full_to_local_dof_id_map_on_refined_mesh,
        dsp);
      mass_matrix_sp.copy_from(dsp);
      mass_matrix.reinit(mass_matrix_sp);

      // Assemble the mass matrix. Only cells within the subdomain are
      // considered.
      assemble_fem_scaled_mass_matrix(
        dof_handler_dual_space,
        dof_handler_primal_space,
        subdomain_material_ids,
        dual_space_dof_selectors_on_refined_mesh,
        dual_space_full_to_local_dof_id_map_on_refined_mesh,
        primal_space_dof_selectors_on_refined_mesh,
        primal_space_full_to_local_dof_id_map_on_refined_mesh,
        1.0,
        quad_rule,
        mass_matrix);
    }
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  build_cluster_and_block_cluster_trees(
    const HMatrixParameters                         &hmat_params,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings)
{
  /**
   * Generate lists of DoF indices.
   */
  const unsigned int n_dofs =
    is_full_domain ? dof_handler_dual_space.n_dofs(1) :
                     dual_space_local_to_full_dof_id_map_on_refined_mesh.size();
  std::vector<types::global_dof_index> dof_indices_in_dual_space(n_dofs);
  gen_linear_indices<vector_uta, types::global_dof_index>(
    dof_indices_in_dual_space);

  /**
   * Get the spatial coordinates of the support points and calculate the average
   * mesh cell size at each support point.
   */
  std::vector<Point<spacedim>> support_points_in_dual_space(n_dofs);
  std::vector<double>          dof_average_cell_size_list(n_dofs, 0.0);
  if (is_full_domain)
    {
      DoFToolsExt::map_mg_dofs_to_support_points(mappings[0]->get_mapping(),
                                                 dof_handler_dual_space,
                                                 1,
                                                 support_points_in_dual_space);
      DoFToolsExt::map_mg_dofs_to_average_cell_size(dof_handler_dual_space,
                                                    1,
                                                    dof_average_cell_size_list);
    }
  else
    {
      DoFToolsExt::map_mg_dofs_to_support_points(
        mappings[0]->get_mapping(),
        dof_handler_dual_space,
        1,
        subdomain_material_ids,
        dual_space_dof_selectors_on_refined_mesh,
        dual_space_full_to_local_dof_id_map_on_refined_mesh,
        support_points_in_dual_space);
      DoFToolsExt::map_mg_dofs_to_average_cell_size(
        dof_handler_dual_space,
        1,
        dual_space_local_to_full_dof_id_map_on_refined_mesh,
        dof_average_cell_size_list);
    }

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
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
template <typename SurfaceNormalDetector>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  build_preconditioner_hmat_on_refined_mesh(
    const unsigned int                               thread_num,
    const HMatrixParameters                         &hmat_params,
    const SubdomainTopology<dim, spacedim>          &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule)
{
  /**
   * Get the internal-to-external DoF numberings.
   */
  std::vector<types::global_dof_index> &dof_i2e_numbering =
    ct.get_internal_to_external_dof_numbering();

  preconditioner_hmat = HMatrix<spacedim, RangeNumberType>(
    bct,
    hmat_params.max_hmat_rank,
    HMatrixSupport::Property::symmetric,
    HMatrixSupport::BlockType::diagonal_block);

  ACAConfig aca_config(hmat_params.max_hmat_rank,
                       hmat_params.aca_relative_error,
                       hmat_params.eta);

  /**
   * When the kernel function is the hyper-singular function, which has a
   * non-trival kernel space, e.g. \f$\mathrm{span}\{1\}\f$, we need to add a
   * stabilization term to the bilinear form. In other cases, stabilization
   * is not needed.
   */
  if (preconditioner_kernel.kernel_type == KernelType::HyperSingularRegular)
    {
      /**
       * Assemble the preconditioning matrix.
       */
      if (is_full_domain)
        {
          // When the preconditioner is on the full domain, the hyper singular
          // bilinear form should be stabilized.

          /**
           * Set natural density as constant vector 1 on each subdomain and set
           * the alpha factor as 1.
           */
          const unsigned int n_subdomains =
            subdomain_topology.get_subdomain_to_surface().size();
          std::vector<Vector<RangeNumberType>> natural_densities(n_subdomains);
          for (auto &vec : natural_densities)
            vec.reinit(dof_handler_primal_space.n_dofs(1));

          assemble_indicator_vectors_for_subdomains(
            dof_handler_primal_space,
            subdomain_topology,
            mappings,
            material_id_to_mapping_index,
            natural_densities);

          const double alpha_for_neumann = 1.0;

          /**
           * Calculate the vector \f$a\f$ in \f$\alpha a a^T\f$, where \f$a\f$
           * is the multiplication of the mass matrix and the natural density.
           *
           * N.B. The mass matrix on the refined mesh maps from the primal
           * function space to the dual space, which is consistent with the
           * vector
           * size of the input vector @p natural_density and the output vector
           * @p mass_vmult_weq.
           */
          std::vector<Vector<RangeNumberType>> mass_vmult_weq(n_subdomains);
          for (auto &vec : mass_vmult_weq)
            vec.reinit(dof_handler_dual_space.n_dofs(1));

          Vector<RangeNumberType> mass_vmult_weq_external_dof_numbering(
            dof_handler_dual_space.n_dofs(1));
          for (unsigned int i = 0; i < n_subdomains; i++)
            {
              mass_matrix.vmult(mass_vmult_weq_external_dof_numbering,
                                natural_densities[i]);
              permute_vector(mass_vmult_weq_external_dof_numbering,
                             dof_i2e_numbering,
                             mass_vmult_weq[i]);
            }

          fill_hmatrix_with_aca_plus_smp(thread_num,
                                         preconditioner_hmat,
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
        // When the preconditioner is on a subdomain, there is no need to
        // stablize the hyper singular bilinear form.
        fill_hmatrix_with_aca_plus_smp(
          thread_num,
          preconditioner_hmat,
          aca_config,
          preconditioner_kernel,
          1.0,
          dof_to_cell_topo_dual_space,
          dof_to_cell_topo_dual_space,
          sauter_quad_rule,
          dof_handler_dual_space,
          dof_handler_dual_space,
          &dual_space_local_to_full_dof_id_map_on_refined_mesh,
          &dual_space_local_to_full_dof_id_map_on_refined_mesh,
          dof_i2e_numbering,
          dof_i2e_numbering,
          mappings,
          material_id_to_mapping_index,
          normal_detector,
          true);
    }
  else
    {
      if (is_full_domain)
        fill_hmatrix_with_aca_plus_smp(thread_num,
                                       preconditioner_hmat,
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
      else
        fill_hmatrix_with_aca_plus_smp(
          thread_num,
          preconditioner_hmat,
          aca_config,
          preconditioner_kernel,
          1.0,
          dof_to_cell_topo_dual_space,
          dof_to_cell_topo_dual_space,
          sauter_quad_rule,
          dof_handler_dual_space,
          dof_handler_dual_space,
          &dual_space_local_to_full_dof_id_map_on_refined_mesh,
          &dual_space_local_to_full_dof_id_map_on_refined_mesh,
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
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  generate_dof_selectors()
{
  // Allocate memory for DoF selectors..
  primal_space_dof_selectors_on_primal_mesh.resize(
    dof_handler_primal_space.n_dofs(0));
  primal_space_dof_selectors_on_refined_mesh.resize(
    dof_handler_primal_space.n_dofs(1));
  dual_space_dof_selectors_on_refined_mesh.resize(
    dof_handler_dual_space.n_dofs(1));

  if (is_subdomain_open && dof_handler_primal_space.get_fe().conforms(
                             FiniteElementData<dim>::Conformity::H1))
    {
      // Select DoFs by excluding those in complement subdomain.
      DoFToolsExt::
        extract_material_domain_mg_dofs_by_excluding_complement_subdomain(
          dof_handler_primal_space,
          0,
          subdomain_complement_material_ids,
          primal_space_dof_selectors_on_primal_mesh);
      DoFToolsExt::
        extract_material_domain_mg_dofs_by_excluding_complement_subdomain(
          dof_handler_primal_space,
          1,
          subdomain_complement_material_ids,
          primal_space_dof_selectors_on_refined_mesh);
    }
  else
    {
      // Select DoFs in the subdomain.
      DoFToolsExt::extract_material_domain_mg_dofs(
        dof_handler_primal_space,
        0,
        subdomain_material_ids,
        primal_space_dof_selectors_on_primal_mesh);
      DoFToolsExt::extract_material_domain_mg_dofs(
        dof_handler_primal_space,
        1,
        subdomain_material_ids,
        primal_space_dof_selectors_on_refined_mesh);
    }

  if (is_subdomain_open && dof_handler_dual_space.get_fe().conforms(
                             FiniteElementData<dim>::Conformity::H1))
    {
      // Select DoFs by excluding those in complement subdomain.
      DoFToolsExt::
        extract_material_domain_mg_dofs_by_excluding_complement_subdomain(
          dof_handler_dual_space,
          1,
          subdomain_complement_material_ids,
          dual_space_dof_selectors_on_refined_mesh);
    }
  else
    {
      // Select DoFs in the subdomain.
      DoFToolsExt::extract_material_domain_mg_dofs(
        dof_handler_dual_space,
        1,
        subdomain_material_ids,
        dual_space_dof_selectors_on_refined_mesh);
    }
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  generate_maps_between_full_and_local_dof_ids()
{
  primal_space_full_to_local_dof_id_map_on_primal_mesh.resize(
    dof_handler_primal_space.n_dofs(0));
  primal_space_local_to_full_dof_id_map_on_primal_mesh.reserve(
    dof_handler_primal_space.n_dofs(0));
  primal_space_full_to_local_dof_id_map_on_refined_mesh.resize(
    dof_handler_primal_space.n_dofs(1));
  primal_space_local_to_full_dof_id_map_on_refined_mesh.reserve(
    dof_handler_primal_space.n_dofs(1));
  dual_space_full_to_local_dof_id_map_on_refined_mesh.resize(
    dof_handler_dual_space.n_dofs(1));
  dual_space_local_to_full_dof_id_map_on_refined_mesh.reserve(
    dof_handler_dual_space.n_dofs(1));

  DoFToolsExt::generate_maps_between_full_and_local_dof_ids(
    primal_space_dof_selectors_on_primal_mesh,
    primal_space_full_to_local_dof_id_map_on_primal_mesh,
    primal_space_local_to_full_dof_id_map_on_primal_mesh);
  DoFToolsExt::generate_maps_between_full_and_local_dof_ids(
    primal_space_dof_selectors_on_refined_mesh,
    primal_space_full_to_local_dof_id_map_on_refined_mesh,
    primal_space_local_to_full_dof_id_map_on_refined_mesh);
  DoFToolsExt::generate_maps_between_full_and_local_dof_ids(
    dual_space_dof_selectors_on_refined_mesh,
    dual_space_full_to_local_dof_id_map_on_refined_mesh,
    dual_space_local_to_full_dof_id_map_on_refined_mesh);
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  initialize_dof_handlers()
{
  // Initialize DoF handlers for primal and dual function spaces.
  dof_handler_primal_space.reinit(tria);
  dof_handler_dual_space.reinit(tria);

  dof_handler_primal_space.distribute_dofs(fe_primal_space);
  dof_handler_primal_space.distribute_mg_dofs();
  dof_handler_dual_space.distribute_dofs(fe_dual_space);
  dof_handler_dual_space.distribute_mg_dofs();
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  build_dof_to_cell_topology()
{
  // Generate DoF-to-cell topologies for the dual function space on the
  // refined mesh.
  cell_iterators_dual_space.reserve(tria.n_cells(1));

  if (is_full_domain)
    {
      for (const auto &cell :
           dof_handler_dual_space.mg_cell_iterators_on_level(1))
        cell_iterators_dual_space.push_back(cell);

      DoFToolsExt::build_mg_dof_to_cell_topology(dof_to_cell_topo_dual_space,
                                                 cell_iterators_dual_space,
                                                 dof_handler_dual_space,
                                                 1);
    }
  else
    {
      if (truncate_function_space_dof_support_within_subdomain)
        {
          for (const auto &cell :
               dof_handler_dual_space.mg_cell_iterators_on_level(1))
            {
              auto found_iter =
                subdomain_material_ids.find(cell->material_id());

              if (found_iter != subdomain_material_ids.end())
                cell_iterators_dual_space.push_back(cell);
            }
        }
      else
        {
          for (const auto &cell :
               dof_handler_dual_space.mg_cell_iterators_on_level(1))
            cell_iterators_dual_space.push_back(cell);
        }

      DoFToolsExt::build_mg_dof_to_cell_topology(
        dof_to_cell_topo_dual_space,
        cell_iterators_dual_space,
        dof_handler_dual_space,
        dual_space_dof_selectors_on_refined_mesh,
        1);
    }
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
template <typename SurfaceNormalDetector>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  setup_preconditioner(
    const unsigned int                               thread_num,
    const HMatrixParameters                         &hmat_params,
    const SubdomainTopology<dim, spacedim>          &subdomain_topology,
    const std::vector<MappingInfo<dim, spacedim> *> &mappings,
    const std::map<types::material_id, unsigned int>
                                    &material_id_to_mapping_index,
    const SurfaceNormalDetector     &normal_detector,
    const SauterQuadratureRule<dim> &sauter_quad_rule,
    const Quadrature<dim>           &quad_rule_for_mass)
{
  initialize_dof_handlers();

  if (!is_full_domain)
    {
      generate_dof_selectors();
      generate_maps_between_full_and_local_dof_ids();
    }

  build_dof_to_cell_topology();
  build_coupling_matrix();
  build_averaging_matrix();
  build_mass_matrix_on_refined_mesh(quad_rule_for_mass);
  build_cluster_and_block_cluster_trees(hmat_params, mappings);
  build_preconditioner_hmat_on_refined_mesh(thread_num,
                                            hmat_params,
                                            subdomain_topology,
                                            mappings,
                                            material_id_to_mapping_index,
                                            normal_detector,
                                            sauter_quad_rule);

  mass_matrix_triple.reinit();
  mass_matrix_transpose_triple.reinit();
  reinit();

  // Compute the diagonal entries of the mass matrix triple, which will be
  // used in the Jacobi precondition.
  compute_mass_matrix_triple_diag_reciprocal();

  // Print out the matrices.
#if ENABLE_PRECONDITIONER_MATRIX_EXPORT == 1
  std::ofstream out_mat(type + std::string("-preconditioner-matrices.dat"));

  print_sparse_matrix_to_mat(out_mat, "Cp", coupling_matrix, 15, true, 25);
  print_sparse_matrix_to_mat(out_mat, "Cd", averaging_matrix, 15, true, 25);
  print_sparse_matrix_to_mat(out_mat, "Mr", mass_matrix, 15, true, 25);
  preconditioner_hmat.print_as_formatted_full_matrix(
    out_mat, "Br", 15, true, 25);
  print_vector_to_mat(out_mat,
                      "M_diag_reciprocal",
                      mass_matrix_triple_diag_reciprocal);
  print_vector_to_mat(out_mat,
                      "primal_space_dof_i2e_numbering",
                      primal_space_dof_i2e_numbering);
  print_vector_to_mat(out_mat,
                      "primal_space_dof_e2i_numbering",
                      primal_space_dof_e2i_numbering);
  print_vector_to_mat(out_mat,
                      "Br_ct_i2e_numbering",
                      ct.get_internal_to_external_dof_numbering());
  print_vector_to_mat(out_mat,
                      "Br_ct_e2i_numbering",
                      ct.get_external_to_internal_dof_numbering());

  out_mat.close();
#endif
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  solve_mass_matrix_triple(Vector<RangeNumberType>       &y,
                           const Vector<RangeNumberType> &x) const
{
  SolverControl solver_control(solve_mass_matrix_max_iter,
                               solve_mass_matrix_tol,
                               solve_mass_matrix_log_history,
                               solve_mass_matrix_log_result);
  SolverGMRES<> solver(solver_control);

  PreconditionJacobi<MassMatrixTriple> precond;
  precond.initialize(
    mass_matrix_triple,
    typename PreconditionJacobi<MassMatrixTriple>::AdditionalData(
      solve_mass_matrix_relaxation));

  solver.solve(mass_matrix_triple, y, x, precond);
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  solve_mass_matrix_transpose_triple(Vector<RangeNumberType>       &y,
                                     const Vector<RangeNumberType> &x) const
{
  SolverControl solver_control(solve_mass_matrix_max_iter,
                               solve_mass_matrix_tol,
                               solve_mass_matrix_log_history,
                               solve_mass_matrix_log_result);
  SolverGMRES<> solver(solver_control);

  PreconditionJacobi<MassMatrixTransposeTriple> precond;
  precond.initialize(
    mass_matrix_transpose_triple,
    typename PreconditionJacobi<MassMatrixTransposeTriple>::AdditionalData(
      solve_mass_matrix_relaxation));

  solver.solve(mass_matrix_transpose_triple, y, x, precond);
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  reinit()
{
  x_external_dof_numbering->reinit(coupling_matrix.m());
  v1->reinit(averaging_matrix.m());
  v2->reinit(averaging_matrix.n());
  v2_internal_dof_numbering->reinit(averaging_matrix.n());
  v3->reinit(averaging_matrix.n());
  v3_external_dof_numbering->reinit(averaging_matrix.n());
  y_external_dof_numbering->reinit(coupling_matrix.m());

  mass_matrix_triple_diag_reciprocal.reinit(averaging_matrix.m());
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  compute_mass_matrix_triple_diag_reciprocal()
{
  Vector<RangeNumberType> Cd_row;
  Vector<RangeNumberType> Cp_row;
  // The result vector of @p Mr*Cp_row.
  Vector<RangeNumberType> v;

  // Iterate over each row.
  for (typename SparseMatrix<RangeNumberType>::size_type i = 0;
       i < averaging_matrix.m();
       i++)
    {
      Cd_row.reinit(averaging_matrix.n());
      Cp_row.reinit(coupling_matrix.n());
      v.reinit(mass_matrix.m());

      // Collect the ith row of Cd.
      for (auto it = averaging_matrix.begin(i); it != averaging_matrix.end(i);
           it++)
        {
          Cd_row(it->column()) = it->value();
        }

      // Collect the ith row of Cp.
      for (auto it = coupling_matrix.begin(i); it != coupling_matrix.end(i);
           it++)
        {
          Cp_row(it->column()) = it->value();
        }

      mass_matrix.vmult(v, Cp_row);
      mass_matrix_triple_diag_reciprocal(i) = 1.0 / (Cd_row * v);
    }
}


template <int dim,
          int spacedim,
          typename KernelFunctionType,
          typename RangeNumberType>
void
OperatorPreconditioner<dim, spacedim, KernelFunctionType, RangeNumberType>::
  vmult(Vector<RangeNumberType> &y, const Vector<RangeNumberType> &x) const
{
  permute_vector(x, primal_space_dof_e2i_numbering, *x_external_dof_numbering);
  solve_mass_matrix_transpose_triple(*v1, *x_external_dof_numbering);
  averaging_matrix.Tvmult(*v2, *v1);
  permute_vector(*v2,
                 ct.get_internal_to_external_dof_numbering(),
                 *v2_internal_dof_numbering);
  preconditioner_hmat.vmult(*v3, *v2_internal_dof_numbering);
  permute_vector(*v3,
                 ct.get_external_to_internal_dof_numbering(),
                 *v3_external_dof_numbering);
  averaging_matrix.vmult(*v1, *v3_external_dof_numbering);
  solve_mass_matrix_triple(*y_external_dof_numbering, *v1);
  permute_vector(*y_external_dof_numbering, primal_space_dof_i2e_numbering, y);
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_PRECONDITIONERS_OPERATOR_PRECONDITIONER_H_