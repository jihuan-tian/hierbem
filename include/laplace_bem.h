/** \file laplace_bem.h
 * \brief Implementation of BEM involving kernel functions and singular
 * numerical quadratures.
 *
 * \ingroup sauter_quadrature
 * \date 2020-11-02
 * \author Jihuan Tian
 */

#ifndef INCLUDE_LAPLACE_BEM_H_
#define INCLUDE_LAPLACE_BEM_H_

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "aca_plus.hcu"
#include "bem_general.h"
#include "bem_tools.hcu"
#include "bem_values.h"
#include "block_cluster_tree.h"
#include "cluster_tree.h"
#include "config.h"
#include "cu_profile.hcu"
#include "debug_tools.hcu"
#include "dof_tools_ext.h"
#include "grid_out_ext.h"
#include "hblockmatrix_skew_symm.h"
#include "hblockmatrix_skew_symm_preconditioner.h"
#include "hmatrix.h"
#include "hmatrix_symm.h"
#include "hmatrix_symm_preconditioner.h"
#include "laplace_kernels.hcu"
#include "mapping_q_generic_ext.h"
#include "quadrature.templates.h"
#include "read_octave_data.h"
#include "triangulation_tools.h"

namespace HierBEM
{
  using namespace dealii;

  template <int dim, int spacedim = dim>
  class LaplaceBEM
  {
  public:
    /**
     * Enum for various types of Laplace problem
     */
    enum ProblemType
    {
      NeumannBCProblem,   //!< NeumannBCProblem
      DirichletBCProblem, //!< DirichletBCProblem
      MixedBCProblem,     //!< MixedBCProblem
      UndefinedProblem
    };

    // Declaration of friend functions.
    template <int dim1,
              int spacedim1,
              typename RangeNumberType,
              typename MatrixType>
    friend void
    assemble_fem_scaled_mass_matrix(
      const DoFHandler<dim1, spacedim1> &dof_handler_for_test_space,
      const DoFHandler<dim1, spacedim1> &dof_handler_for_trial_space,
      const RangeNumberType              factor,
      const Quadrature<dim1>            &quad_rule,
      MatrixType                        &target_full_matrix);

    template <int dim1,
              int spacedim1,
              template <int, typename>
              typename KernelFunctionType,
              typename RangeNumberType,
              typename MatrixType>
    friend void
    assemble_bem_full_matrix(
      const KernelFunctionType<spacedim, RangeNumberType> &kernel,
      const DoFHandler<dim1, spacedim1>   &dof_handler_for_test_space,
      const DoFHandler<dim1, spacedim1>   &dof_handler_for_trial_space,
      MappingQGenericExt<dim1, spacedim1> &kx_mapping,
      MappingQGenericExt<dim1, spacedim1> &ky_mapping,
      typename MappingQGeneric<dim1, spacedim1>::InternalData &kx_mapping_data,
      typename MappingQGeneric<dim1, spacedim1>::InternalData &ky_mapping_data,
      const std::map<typename Triangulation<dim1, spacedim1>::cell_iterator,
                     typename Triangulation<dim1 + 1, spacedim1>::face_iterator>
        &map_from_test_space_mesh_to_volume_mesh,
      const std::map<typename Triangulation<dim1, spacedim1>::cell_iterator,
                     typename Triangulation<dim1 + 1, spacedim1>::face_iterator>
        &map_from_trial_space_mesh_to_volume_mesh,
      const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
      const SauterQuadratureRule<dim1>     &sauter_quad_rule,
      MatrixType                           &target_full_matrix);


    /**
     * The large integer for shifting the material id in the interfacial domain
     * \f$\Omega_I\f$.
     */
    const static types::material_id material_id_shift_for_interfacial_domain;

    /**
     * Default constructor
     */
    LaplaceBEM();

    /**
     * Constructor for solving Laplace problem using full matrix, which has only
     * verification purpose.
     *
     * \comment{Here we do not initialize the functions describing boundary
     * conditions, because which one of them should be initialized depends on
     * the problem type. This task is transferred to the user code.}
     */
    LaplaceBEM(unsigned int fe_order_for_dirichlet_space,
               unsigned int fe_order_for_neumann_space,
               unsigned int mapping_order_for_dirichlet_domain,
               unsigned int mapping_order_for_neumann_domain,
               ProblemType  problem_type,
               bool         is_interior_problem,
               unsigned int thread_num);

    /**
     * Constructor for solving Laplace problem using \hmatrix.
     *
     * @param mesh_file_name
     * @param fe_order_for_dirichlet_space
     * @param fe_order_for_neumann_space
     * @param mapping_order
     * @param problem_type
     * @param n_min_for_ct
     * @param n_min_for_bct
     * @param eta
     * @param max_hmat_rank
     * @param aca_relative_error
     * @param thread_num
     */
    LaplaceBEM(unsigned int fe_order_for_dirichlet_space,
               unsigned int fe_order_for_neumann_space,
               unsigned int mapping_order_for_dirichlet_domain,
               unsigned int mapping_order_for_neumann_domain,
               ProblemType  problem_type,
               bool         is_interior_problem,
               unsigned int n_min_for_ct,
               unsigned int n_min_for_bct,
               double       eta,
               unsigned int max_hmat_rank,
               double       aca_relative_error,
               double       eta_for_preconditioner,
               unsigned int max_hmat_rank_for_preconditioner,
               double       aca_relative_error_for_preconditioner,
               unsigned int thread_num);

    /**
     * Destructor, where DoF handlers are cleared.
     */
    ~LaplaceBEM();

    /**
     * Read the volume mesh from a file.
     */
    void
    read_volume_mesh(const std::string &mesh_file);

    /**
     * Read independent mesh files for two spheres then merge the triangulation.
     */
    void
    read_two_spheres();

    /**
     * Generate volume mesh for the two spheres.
     */
    void
    generate_volume_mesh();

    void
    set_dirichlet_boundary_ids(std::initializer_list<types::boundary_id> ilist);

    void
    set_neumann_boundary_ids(std::initializer_list<types::boundary_id> ilist);

    /**
     * Prepare for matrix assembly, which includes:
     * . initialization of DoF handlers
     * . memory allocation for matrices
     */
    void
    setup_system();

    /**
     * Assign Dirichlet boundary condition function object.
     *
     * @param functor_ptr
     */
    void
    assign_dirichlet_bc(Function<spacedim> &functor);

    /**
     * Assign Neumann boundary condition function object.
     *
     * @param functor_ptr
     */
    void
    assign_neumann_bc(Function<spacedim> &functor);

    /**
     * Assemble full matrix system, which is only for verification purpose.
     */
    void
    assemble_full_matrix_system();

    /**
     * Assemble \hmatrix system.
     */
    void
    assemble_hmatrix_system();

    /**
     * Assemble \hmatrix preconditioner.
     */
    void
    assemble_hmatrix_preconditioner();

    void
    solve();

    void
    output_results() const;

    void
    output_potential_at_target_points() const;

    void
    run();

    /**
     * Verify if the calculated Neumann solution is in the space
     * \f$H^{1/2}_*(\Gamma)\f$.
     */
    void
    verify_neumann_solution_in_space();

    double
    get_alpha_for_neumann() const
    {
      return alpha_for_neumann;
    }

    void
    set_alpha_for_neumann(double alphaForNeumann)
    {
      alpha_for_neumann = alphaForNeumann;
    }

  private:
    void
    generate_cell_iterators();

    /**
     * Initialize the mapping data object.
     */
    void
    initialize_mapping_data();

    /**
     * Solve the equation \f$Vw_{\rm eq}=1\f$ for the natural density \f$w_{\rm
     * eq}\f$ and calculate the stabilization factor \f$\alpha\f$.
     */
    void
    solve_natural_density();

    /**
     * Finite element order for the Dirichlet space.
     */
    unsigned int fe_order_for_dirichlet_space;

    /**
     * Finite element order for the Neumann space.
     */
    unsigned int fe_order_for_neumann_space;

    /**
     * Laplace problem type to be solved.
     */
    ProblemType problem_type;

    /**
     * Whether the problem is interior or exterior.
     */
    bool is_interior_problem;

    /**
     * Number of threads
     */
    unsigned int thread_num;

    /**
     * Triangulation for the volume mesh.
     */
    Triangulation<dim + 1, spacedim> volume_triangulation;

    /*
     * Triangulation for the surface mesh.
     */
    Triangulation<dim, spacedim> surface_triangulation;

    /**
     * Map from cell iterators in the surface mesh to the face iterators in the
     * original volume mesh.
     */
    std::map<typename Triangulation<dim, spacedim>::cell_iterator,
             typename Triangulation<dim + 1, spacedim>::face_iterator>
      map_from_surface_mesh_to_volume_mesh;

    /**
     * A set of boundary indices for the Dirichlet domain.
     */
    std::set<types::boundary_id> boundary_ids_for_dirichlet_domain;

    /**
     * A set of boundary indices for the extended Dirichlet domain
     * \f$\tilde{\Gamma}_D\f$.
     */
    std::set<types::boundary_id> boundary_ids_for_extended_dirichlet_domain;

    /**
     * A set of boundary indices for the Neumann domain.
     */
    std::set<types::boundary_id> boundary_ids_for_neumann_domain;

    /**
     * Finite element \f$H^{\frac{1}{2}+s}\f$ for the Dirichlet space. At
     * present, it is implemented as a continuous Lagrange space.
     */
    FE_Q<dim, spacedim> fe_for_dirichlet_space;
    /**
     * Finite element \f$H^{-\frac{1}{2}+s}\f$ for the Neumann space. At
     * present, it is implemented as a discontinuous Lagrange space.
     */
    FE_DGQ<dim, spacedim> fe_for_neumann_space;

    /**
     * Definition of DoFHandlers for a series of combination of finite element
     * spaces and triangulations.
     */
    DoFHandler<dim, spacedim> dof_handler_for_dirichlet_space;
    DoFHandler<dim, spacedim> dof_handler_for_neumann_space;

    /**
     * Map from the selected DoF indices to the complete indices in the DoF
     * handler for the Dirichlet space. Only effective
     * in the mixed boundary value problem.
     */
    std::vector<types::global_dof_index>
      local_to_full_dirichlet_dof_indices_on_dirichlet_domain;

    /**
     * Map from the selected DoF indices to the complete indices in the DoF
     * handler for the Dirichlet space. Only effective
     * in the mixed boundary value problem.
     */
    std::vector<types::global_dof_index>
      local_to_full_dirichlet_dof_indices_on_neumann_domain;

    std::vector<types::global_dof_index>
      local_to_full_neumann_dof_indices_on_dirichlet_domain;

    std::vector<types::global_dof_index>
      local_to_full_neumann_dof_indices_on_neumann_domain;

    /**
     * A list of Boolean flags indicating if each DoF is selected in the
     * Dirichlet space on the Dirichlet domain.
     */
    std::vector<bool> dof_selectors_for_dirichlet_space_on_dirichlet_domain;

    /**
     * A list of Boolean flags indicating if each DoF is selected in the
     * Dirichlet space on the Neumann domain.
     */
    std::vector<bool> dof_selectors_for_dirichlet_space_on_neumann_domain;

    std::vector<bool> dof_selectors_for_neumann_space_on_dirichlet_domain;

    std::vector<bool> dof_selectors_for_neumann_space_on_neumann_domain;

    /**
     * Map from external DoF indices to internal indices and from internal
     * indices to external indices.
     */
    const std::vector<types::global_dof_index>
      *dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain;
    const std::vector<types::global_dof_index>
      *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain;
    const std::vector<types::global_dof_index>
      *dof_e2i_numbering_for_dirichlet_space_on_neumann_domain;
    const std::vector<types::global_dof_index>
      *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain;
    const std::vector<types::global_dof_index>
      *dof_e2i_numbering_for_neumann_space_on_dirichlet_domain;
    const std::vector<types::global_dof_index>
      *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain;
    const std::vector<types::global_dof_index>
      *dof_e2i_numbering_for_neumann_space_on_neumann_domain;
    const std::vector<types::global_dof_index>
      *dof_i2e_numbering_for_neumann_space_on_neumann_domain;

    std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
      cell_iterators_for_dirichlet_space;
    std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
      cell_iterators_for_neumann_space;

    /**
     * DoF-to-cell topologies for various DoF handlers, which are used for
     * matrix assembly on a pair of DoFs.
     */
    DofToCellTopology<dim, spacedim> dof_to_cell_topo_for_dirichlet_space;
    DofToCellTopology<dim, spacedim> dof_to_cell_topo_for_neumann_space;

    /**
     * Polynomial order for describing the geometric mapping for the Dirichlet
     * domain, i.e. the transformation from the unit cell to a real cell.
     */
    unsigned int mapping_order_for_dirichlet_domain;
    /**
     * Polynomial order for describing the geometric mapping for the Neumann
     * domain, i.e. the transformation from the unit cell to a real cell.
     */
    unsigned int mapping_order_for_neumann_domain;
    /**
     * Geometric mapping object for the Dirichlet domain.
     */
    MappingQGenericExt<dim, spacedim> kx_mapping_for_dirichlet_domain;
    MappingQGenericExt<dim, spacedim> ky_mapping_for_dirichlet_domain;
    /**
     * Geometric mapping object for the Neumann domain.
     */
    MappingQGenericExt<dim, spacedim> kx_mapping_for_neumann_domain;
    MappingQGenericExt<dim, spacedim> ky_mapping_for_neumann_domain;
    /**
     * Pointer to the internal data held by the @p Mapping object for the
     * Dirichlet domain.
     */
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
      kx_mapping_data_for_dirichlet_domain;
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
      ky_mapping_data_for_dirichlet_domain;
    /**
     * Pointer to the internal data held by the @p Mapping object for the
     * Neumann domain.
     */
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
      kx_mapping_data_for_neumann_domain;
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
      ky_mapping_data_for_neumann_domain;

    /**
     * Kernel function for the single layer potential.
     */
    HierBEM::CUDAWrappers::LaplaceKernel::SingleLayerKernel<3>
      single_layer_kernel;
    /**
     * Kernel function for the double layer potential.
     */
    HierBEM::CUDAWrappers::LaplaceKernel::DoubleLayerKernel<3>
      double_layer_kernel;
    /**
     * Kernel function for the adjoint double layer potential.
     */
    HierBEM::CUDAWrappers::LaplaceKernel::AdjointDoubleLayerKernel<3>
      adjoint_double_layer_kernel;
    /**
     * Kernel function for the hyper-singular potential.
     */
    HierBEM::CUDAWrappers::LaplaceKernel::HyperSingularKernelRegular<3>
      hyper_singular_kernel;

    /**
     * Full matrices for verification purpose.
     */
    FullMatrix<double> V1_matrix;
    FullMatrix<double> K1_matrix;
    FullMatrix<double> K_prime1_matrix;
    FullMatrix<double> D1_matrix;
    FullMatrix<double> K2_matrix_with_mass_matrix;
    FullMatrix<double> V2_matrix;
    FullMatrix<double> D2_matrix;
    FullMatrix<double> K_prime2_matrix_with_mass_matrix;

    /**
     * Whether \hmatrix is used.
     */
    bool is_use_hmat;

    /**
     * Cluster trees
     */
    ClusterTree<spacedim> ct_for_dirichlet_space_on_dirichlet_domain;
    ClusterTree<spacedim> ct_for_neumann_space_on_dirichlet_domain;
    ClusterTree<spacedim> ct_for_dirichlet_space_on_neumann_domain;
    ClusterTree<spacedim> ct_for_neumann_space_on_neumann_domain;

    /**
     * Block cluster trees corresponding to discretized bilinear forms in
     * the mixed boundary value problem, which contain all possible cases.
     * \f[
     * \begin{equation}
     * \begin{pmatrix}
     *  -\mathscr{V} & \mathscr{K} \\ \mathscr{K}' & \mathscr{D} \end{pmatrix}
     * \begin{pmatrix}
     *  t \big\vert_{\Gamma_{\rm D}} \\ u \big\vert_{\Gamma_{\rm N}}
     * \end{pmatrix}= \begin{pmatrix}
     *  -\frac{1}{2}\mathscr{I} - \mathscr{K} & \mathscr{V} \\ \mathscr{-D} &
     * \frac{1}{2}\mathscr{I} - \mathscr{K}' \end{pmatrix} \begin{pmatrix} g_D
     * \\ g_N \end{pmatrix} \end{equation} \f]
     * @p V1, @p K1, @p K_prime1 and @p D1 correspond to matrix blocks on the
     * left. @p K2, @p V2, @p D2 and @p K_prime2 correspond to matrix blocks
     * on the right.
     */
    BlockClusterTree<spacedim> bct_for_bilinear_form_V1;
    BlockClusterTree<spacedim> bct_for_bilinear_form_K1;
    BlockClusterTree<spacedim> bct_for_bilinear_form_K_prime1;
    BlockClusterTree<spacedim> bct_for_bilinear_form_D1;
    BlockClusterTree<spacedim> bct_for_bilinear_form_V2;
    BlockClusterTree<spacedim> bct_for_bilinear_form_K2;
    BlockClusterTree<spacedim> bct_for_bilinear_form_K_prime2;
    BlockClusterTree<spacedim> bct_for_bilinear_form_D2;

    /**
     * \hmatrices corresponding to discretized bilinear forms in the
     * mixed boundary value problem, which contain all possible cases.
     */
    HMatrixSymm<spacedim>          V1_hmat;
    HMatrix<spacedim>              K1_hmat;
    HMatrix<spacedim>              K_prime1_hmat;
    HMatrixSymm<spacedim>          D1_hmat;
    HMatrix<spacedim>              V2_hmat;
    HMatrix<spacedim>              K2_hmat_with_mass_matrix;
    HMatrix<spacedim>              K_prime2_hmat_with_mass_matrix;
    HMatrix<spacedim>              D2_hmat;
    HBlockMatrixSkewSymm<spacedim> M_hmat;

    /**
     * Preconditioners
     */
    HMatrixSymmPreconditioner<spacedim>          V1_hmat_preconditioner;
    HMatrixSymmPreconditioner<spacedim>          D1_hmat_preconditioner;
    HMatrix<spacedim>                            M11_in_preconditioner;
    HMatrix<spacedim>                            M12_in_preconditioner;
    HMatrix<spacedim>                            M22_in_preconditioner;
    HBlockMatrixSkewSymmPreconditioner<spacedim> M_hmat_preconditioner;

    /**
     * The sequence of all DoF indices with the values \f$0, 1, \cdots\f$ for
     * different DoFHandlers.
     */
    std::vector<types::global_dof_index>
      dof_indices_for_dirichlet_space_on_dirichlet_domain;
    std::vector<types::global_dof_index>
      dof_indices_for_neumann_space_on_dirichlet_domain;
    std::vector<types::global_dof_index>
      dof_indices_for_dirichlet_space_on_neumann_domain;
    std::vector<types::global_dof_index>
      dof_indices_for_neumann_space_on_neumann_domain;

    /**
     * The list of all support points associated with @p dof_indices held
     * within different DoF handlers.
     */
    std::vector<Point<spacedim>>
      support_points_for_dirichlet_space_on_dirichlet_domain;
    std::vector<Point<spacedim>>
      support_points_for_dirichlet_space_on_neumann_domain;
    std::vector<Point<spacedim>>
      support_points_for_neumann_space_on_dirichlet_domain;
    std::vector<Point<spacedim>>
      support_points_for_neumann_space_on_neumann_domain;

    /**
     * Estimated average cell size values associated with @p dof_indices held
     * within different DoF handlers.
     */
    std::vector<double>
      dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain;
    std::vector<double>
      dof_average_cell_size_for_neumann_space_on_dirichlet_domain;
    std::vector<double>
      dof_average_cell_size_for_dirichlet_space_on_neumann_domain;
    std::vector<double>
      dof_average_cell_size_for_neumann_space_on_neumann_domain;

    /**
     * Minimum cluster size. At present, assume all \bcts share this same
     * parameter.
     */
    unsigned int n_min_for_ct;
    /**
     * Minimum block cluster size. At present, assume all \bcts share this
     * same parameter.
     */
    unsigned int n_min_for_bct;
    /**
     * Admissibility constant. At present, assume all \bcts share this same
     * parameter.
     */
    double eta;
    /**
     * Maximum rank of the \hmatrices to be built. At present, assume all
     * \hmatrices share this same parameter.
     */
    unsigned int max_hmat_rank;
    /**
     * Relative approximation error used in ACA+. At present, assume all
     * \hmatrices share this same parameter.
     */
    double aca_relative_error;
    /**
     * Admissibility constant for the preconditioner.
     */
    double eta_for_preconditioner;
    /**
     * Maximum rank of the \hmatrices to be built for the preconditioner.
     */
    unsigned int max_hmat_rank_for_preconditioner;
    /**
     * Relative approximation error used in ACA+ for the preconditioner.
     */
    double aca_relative_error_for_preconditioner;

    /**
     * Pointer to the Neumann boundary condition function object.
     */
    Function<spacedim> *neumann_bc_functor_ptr;

    /**
     * Neumann boundary condition data on all DoFs in the associated DoF
     * handler. When in the \hmat version, they are in the external DoF
     * numbering.
     */
    Vector<double> neumann_bc;
    Vector<double> neumann_bc_on_selected_dofs;
    /**
     * Neumann boundary condition data on all DoFs in the associated DoF handler
     * in the internal DoF numbering. Only used in the \hmat version.
     */
    Vector<double> neumann_bc_internal_dof_numbering;

    /**
     * The free parameter \f$\alpha\f$ in the variational formulation of the
     * Laplace problem with Neumann boundary condition.
     */
    double alpha_for_neumann;

    /**
     * Pointer to the Dirichlet boundary condition function object.
     */
    Function<spacedim> *dirichlet_bc_functor_ptr;

    /**
     * Dirichlet boundary condition data on all DoFs in the associated DoF
     * handler.
     */
    Vector<double> dirichlet_bc;
    /**
     * Dirichlet boundary condition data on those selected DoFs in the
     * associated DoF handler. When in the \hamt version, they are in the
     * external DoF numbering.
     */
    Vector<double> dirichlet_bc_on_selected_dofs;
    /**
     * Dirichlet boundary condition data on those selected DoFs in the
     * associated DoF handler in the internal DoF numbering. Only used in the
     * \hmat version.
     */
    Vector<double> dirichlet_bc_internal_dof_numbering;

    /**
     * Right hand side vector on the Dirichlet domain. When in the \hmat
     * version, it is in the internal DoF numbering.
     */
    Vector<double> system_rhs_on_dirichlet_domain;
    /**
     * Right hand side vector on the Neumann domain. When in the \hmat version,
     * it is in the internal DoF numbering.
     */
    Vector<double> system_rhs_on_neumann_domain;
    /**
     * Right hand side vector on both the Dirichlet domain and Neumann domain.
     * Only used in the mixed boundary value problem.
     */
    Vector<double> system_rhs_on_combined_domain;

    /**
     * Natural density \f$w_{rm eq}\f$, which is an intermediate solution used
     * in the Neumann boundary problem.
     */
    Vector<double> natural_density;

    /**
     * The result vector for the multiplication of the mass matrix and the
     * natural density \f$w_{\rm eq}\f$
     */
    Vector<double> mass_vmult_weq;

    /**
     * Right hand side vector, which is used in the equation \f$\mathscr{V}
     * w_{\rm eq} = 1\f$ for solving the natural density \f$w_{\rm eq}\f$. Each
     * component in this vector is actually an integration of each test
     * function.
     */
    Vector<double> system_rhs_for_natural_density;

    /**
     * Numerical solution on the Dirichlet domain for full matrix version or in
     * the external DoF numbering for the \hmat version.
     */
    Vector<double> neumann_data;
    /**
     * Numerical solution on the Dirichlet domain in the internal DoF numbering.
     * Only used in the \hmat version.
     */
    Vector<double> neumann_data_on_dirichlet_domain_internal_dof_numbering;
    /**
     * Numerical solution on the Neumann domain on all DoFs in the associated
     * DoF handler.
     */
    Vector<double> dirichlet_data;
    /**
     * Numerical solution on the Neumann domain only on those selected DoFs in
     * the associated DoF handler. When in the \hmat version, they are in the
     * external DoF numbering.
     */
    Vector<double> dirichlet_data_on_neumann_domain;
    Vector<double> neumann_data_on_dirichlet_domain;
    /**
     * Numerical solution on the Neumann domain only on those selected DoFs in
     * the associated DoF handler in the internal DoF numbering. Only used in
     * the \hmat version.
     */
    Vector<double> dirichlet_data_on_neumann_domain_internal_dof_numbering;
    /**
     * Solution vector on both the Dirichlet domain and Neumann domain. Only
     * used in the mixed boundary value problem and in the \hmat version.
     */
    Vector<double> solution_on_combined_domain_internal_dof_numbering;

    /**
     * Analytical solution on the Dirichlet domain.
     */
    Vector<double> analytical_solution_on_dirichlet_domain;
    /**
     * Analytical solution on the Neumann domain.
     */
    Vector<double> analytical_solution_on_neumann_domain;
  };


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::LaplaceBEM()
    : fe_order_for_dirichlet_space(0)
    , fe_order_for_neumann_space(0)
    , problem_type(UndefinedProblem)
    , is_interior_problem(true)
    , thread_num(0)
    , fe_for_dirichlet_space(0)
    , fe_for_neumann_space(0)
    , dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain(nullptr)
    , dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain(nullptr)
    , dof_e2i_numbering_for_dirichlet_space_on_neumann_domain(nullptr)
    , dof_i2e_numbering_for_dirichlet_space_on_neumann_domain(nullptr)
    , dof_e2i_numbering_for_neumann_space_on_dirichlet_domain(nullptr)
    , dof_i2e_numbering_for_neumann_space_on_dirichlet_domain(nullptr)
    , dof_e2i_numbering_for_neumann_space_on_neumann_domain(nullptr)
    , dof_i2e_numbering_for_neumann_space_on_neumann_domain(nullptr)
    , mapping_order_for_dirichlet_domain(0)
    , mapping_order_for_neumann_domain(0)
    , kx_mapping_for_dirichlet_domain(0)
    , ky_mapping_for_dirichlet_domain(0)
    , kx_mapping_for_neumann_domain(0)
    , ky_mapping_for_neumann_domain(0)
    , kx_mapping_data_for_dirichlet_domain(nullptr)
    , ky_mapping_data_for_dirichlet_domain(nullptr)
    , kx_mapping_data_for_neumann_domain(nullptr)
    , ky_mapping_data_for_neumann_domain(nullptr)
    , is_use_hmat(false)
    , n_min_for_ct(0)
    , n_min_for_bct(0) // By default, it is the same as the @p n_min_for_ct
    , eta(0)
    , max_hmat_rank(0)
    , aca_relative_error(0)
    , eta_for_preconditioner(0)
    , max_hmat_rank_for_preconditioner(0)
    , aca_relative_error_for_preconditioner(0)
    , neumann_bc_functor_ptr(nullptr)
    , alpha_for_neumann(1.0)
    , dirichlet_bc_functor_ptr(nullptr)
  {}


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::LaplaceBEM(
    unsigned int fe_order_for_dirichlet_space,
    unsigned int fe_order_for_neumann_space,
    unsigned int mapping_order_for_dirichlet_domain,
    unsigned int mapping_order_for_neumann_domain,
    ProblemType  problem_type,
    bool         is_interior_problem,
    unsigned int thread_num)
    : fe_order_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_order_for_neumann_space(fe_order_for_neumann_space)
    , problem_type(problem_type)
    , is_interior_problem(is_interior_problem)
    , thread_num(thread_num)
    , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_for_neumann_space(fe_order_for_neumann_space)
    , dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain(nullptr)
    , dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain(nullptr)
    , dof_e2i_numbering_for_dirichlet_space_on_neumann_domain(nullptr)
    , dof_i2e_numbering_for_dirichlet_space_on_neumann_domain(nullptr)
    , dof_e2i_numbering_for_neumann_space_on_dirichlet_domain(nullptr)
    , dof_i2e_numbering_for_neumann_space_on_dirichlet_domain(nullptr)
    , dof_e2i_numbering_for_neumann_space_on_neumann_domain(nullptr)
    , dof_i2e_numbering_for_neumann_space_on_neumann_domain(nullptr)
    , mapping_order_for_dirichlet_domain(mapping_order_for_dirichlet_domain)
    , mapping_order_for_neumann_domain(mapping_order_for_neumann_domain)
    , kx_mapping_for_dirichlet_domain(mapping_order_for_dirichlet_domain)
    , ky_mapping_for_dirichlet_domain(mapping_order_for_dirichlet_domain)
    , kx_mapping_for_neumann_domain(mapping_order_for_neumann_domain)
    , ky_mapping_for_neumann_domain(mapping_order_for_neumann_domain)
    , kx_mapping_data_for_dirichlet_domain(nullptr)
    , ky_mapping_data_for_dirichlet_domain(nullptr)
    , kx_mapping_data_for_neumann_domain(nullptr)
    , ky_mapping_data_for_neumann_domain(nullptr)
    , is_use_hmat(false)
    , n_min_for_ct(0)
    , n_min_for_bct(0)
    , eta(0)
    , max_hmat_rank(0)
    , aca_relative_error(0)
    , eta_for_preconditioner(0)
    , max_hmat_rank_for_preconditioner(0)
    , aca_relative_error_for_preconditioner(0)
    , neumann_bc_functor_ptr(nullptr)
    , alpha_for_neumann(1.0)
    , dirichlet_bc_functor_ptr(nullptr)
  {
    initialize_mapping_data();
  }


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::LaplaceBEM(
    unsigned int fe_order_for_dirichlet_space,
    unsigned int fe_order_for_neumann_space,
    unsigned int mapping_order_for_dirichlet_domain,
    unsigned int mapping_order_for_neumann_domain,
    ProblemType  problem_type,
    bool         is_interior_problem,
    unsigned int n_min_for_ct,
    unsigned int n_min_for_bct,
    double       eta,
    unsigned int max_hmat_rank,
    double       aca_relative_error,
    double       eta_for_preconditioner,
    unsigned int max_hmat_rank_for_preconditioner,
    double       aca_relative_error_for_preconditioner,
    unsigned int thread_num)
    : fe_order_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_order_for_neumann_space(fe_order_for_neumann_space)
    , problem_type(problem_type)
    , is_interior_problem(is_interior_problem)
    , thread_num(thread_num)
    , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_for_neumann_space(fe_order_for_neumann_space)
    , dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain(nullptr)
    , dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain(nullptr)
    , dof_e2i_numbering_for_dirichlet_space_on_neumann_domain(nullptr)
    , dof_i2e_numbering_for_dirichlet_space_on_neumann_domain(nullptr)
    , dof_e2i_numbering_for_neumann_space_on_dirichlet_domain(nullptr)
    , dof_i2e_numbering_for_neumann_space_on_dirichlet_domain(nullptr)
    , dof_e2i_numbering_for_neumann_space_on_neumann_domain(nullptr)
    , dof_i2e_numbering_for_neumann_space_on_neumann_domain(nullptr)
    , mapping_order_for_dirichlet_domain(mapping_order_for_dirichlet_domain)
    , mapping_order_for_neumann_domain(mapping_order_for_neumann_domain)
    , kx_mapping_for_dirichlet_domain(mapping_order_for_dirichlet_domain)
    , ky_mapping_for_dirichlet_domain(mapping_order_for_dirichlet_domain)
    , kx_mapping_for_neumann_domain(mapping_order_for_neumann_domain)
    , ky_mapping_for_neumann_domain(mapping_order_for_neumann_domain)
    , kx_mapping_data_for_dirichlet_domain(nullptr)
    , ky_mapping_data_for_dirichlet_domain(nullptr)
    , kx_mapping_data_for_neumann_domain(nullptr)
    , ky_mapping_data_for_neumann_domain(nullptr)
    , is_use_hmat(true)
    , n_min_for_ct(n_min_for_ct)
    , n_min_for_bct(n_min_for_bct)
    , eta(eta)
    , max_hmat_rank(max_hmat_rank)
    , aca_relative_error(aca_relative_error)
    , eta_for_preconditioner(eta_for_preconditioner)
    , max_hmat_rank_for_preconditioner(max_hmat_rank_for_preconditioner)
    , aca_relative_error_for_preconditioner(
        aca_relative_error_for_preconditioner)
    , neumann_bc_functor_ptr(nullptr)
    , alpha_for_neumann(1.0)
    , dirichlet_bc_functor_ptr(nullptr)
  {
    initialize_mapping_data();
  }


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::~LaplaceBEM()
  {
    dof_handler_for_dirichlet_space.clear();
    dof_handler_for_neumann_space.clear();

    map_from_surface_mesh_to_volume_mesh.clear();

    boundary_ids_for_dirichlet_domain.clear();
    boundary_ids_for_extended_dirichlet_domain.clear();
    boundary_ids_for_neumann_domain.clear();

    local_to_full_dirichlet_dof_indices_on_dirichlet_domain.clear();
    local_to_full_dirichlet_dof_indices_on_neumann_domain.clear();
    local_to_full_neumann_dof_indices_on_dirichlet_domain.clear();
    local_to_full_dirichlet_dof_indices_on_neumann_domain.clear();

    neumann_bc_functor_ptr   = nullptr;
    dirichlet_bc_functor_ptr = nullptr;

    dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain = nullptr;
    dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain = nullptr;
    dof_e2i_numbering_for_dirichlet_space_on_neumann_domain   = nullptr;
    dof_i2e_numbering_for_dirichlet_space_on_neumann_domain   = nullptr;
    dof_e2i_numbering_for_neumann_space_on_dirichlet_domain   = nullptr;
    dof_i2e_numbering_for_neumann_space_on_dirichlet_domain   = nullptr;
    dof_e2i_numbering_for_neumann_space_on_neumann_domain     = nullptr;
    dof_i2e_numbering_for_neumann_space_on_neumann_domain     = nullptr;
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::read_volume_mesh(const std::string &mesh_file)
  {
    /**
     * The template parameters @p dim and @p spacedim of the @p LaplaceBEM
     * class are used to describe the surface mesh, and here the volume mesh is
     * to be read, therefore, the dimensions for @p GridIn is
     * <code>(dim+1,spacedim)</code>.
     */
    GridIn<dim + 1, spacedim> grid_in;
    grid_in.attach_triangulation(volume_triangulation);
    std::fstream in(mesh_file);
    grid_in.read_msh(in);
    in.close();

    std::cout << "=== Volume mesh information ===" << std::endl;
    print_mesh_info(std::cout, volume_triangulation);

    map_from_surface_mesh_to_volume_mesh =
      GridGenerator::extract_boundary_mesh(volume_triangulation,
                                           surface_triangulation);

    std::cout << "=== Surface mesh information ===" << std::endl;
    print_mesh_info(std::cout, surface_triangulation);

    /**
     * @internal Save the surface mesh file.
     */
    std::ofstream surface_mesh_file(
      mesh_file.substr(0, mesh_file.find_last_of('.')) + "-surface.msh");
    write_msh_correct(surface_triangulation, surface_mesh_file);
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::read_two_spheres()
  {
    GridIn<spacedim>        grid_in;
    Triangulation<spacedim> left_ball, right_ball;
    grid_in.attach_triangulation(left_ball);
    std::fstream in("left-sphere_hex.msh");
    grid_in.read_msh(in);
    in.close();

    grid_in.attach_triangulation(right_ball);
    in.open("right-sphere_hex.msh");
    grid_in.read_msh(in);
    in.close();

    GridGenerator::merge_triangulations(
      left_ball, right_ball, volume_triangulation, 1e-12, true);

    std::cout << "=== Volume mesh information ===" << std::endl;
    print_mesh_info(std::cout, volume_triangulation);

    map_from_surface_mesh_to_volume_mesh =
      GridGenerator::extract_boundary_mesh(volume_triangulation,
                                           surface_triangulation);

    std::cout << "=== Surface mesh information ===" << std::endl;
    print_mesh_info(std::cout, surface_triangulation);

    /**
     * @internal Save the surface mesh file.
     */
    std::ofstream surface_mesh_file("two-spheres_hex-surface.msh");
    write_msh_correct(surface_triangulation, surface_mesh_file);
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::generate_volume_mesh()
  {
    Triangulation<spacedim> left_ball, right_ball;
    GridGenerator::hyper_ball(left_ball, Point<spacedim>(-1.5, 0, 0), 1);
    GridGenerator::hyper_ball(right_ball, Point<spacedim>(1.5, 0, 0), 1);

    /**
     * @internal Set different manifold ids and material ids to all the cells in
     * the two balls.
     */
    for (typename Triangulation<spacedim>::active_cell_iterator cell =
           left_ball.begin_active();
         cell != left_ball.end();
         cell++)
      {
        cell->set_all_manifold_ids(0);
        cell->set_material_id(0);
      }

    for (typename Triangulation<spacedim>::active_cell_iterator cell =
           right_ball.begin_active();
         cell != right_ball.end();
         cell++)
      {
        cell->set_all_manifold_ids(1);
        cell->set_material_id(1);
      }

    /**
     * @internal @p merge_triangulation can only operate on coarse mesh, i.e.
     * triangulations not refined. During the merging, the material ids are
     * copied. When the last argument is true, the manifold ids are copied.
     * Boundary ids will not be copied.
     */
    GridGenerator::merge_triangulations(
      left_ball, right_ball, volume_triangulation, 1e-12, true);

    /**
     * @internal Assign manifold objects to the two balls in the merged mesh.
     */
    const SphericalManifold<spacedim> left_ball_manifold(
      Point<spacedim>(-1.5, 0, 0));
    const SphericalManifold<spacedim> right_ball_manifold(
      Point<spacedim>(1.5, 0, 0));

    volume_triangulation.set_manifold(0, left_ball_manifold);
    volume_triangulation.set_manifold(1, right_ball_manifold);

    /**
     * Check the material id for each cell in the merged volume mesh. It is
     * verified that the material ids are preserved during the merging of
     * triangulations.
     *
     * Meanwhile, set the boundary id according to the material id. In the
     * further surface mesh extraction, the boundary ids will be automatically
     * converted to the material ids of the cells in the surface mesh.
     */
    for (typename Triangulation<spacedim>::active_cell_iterator cell =
           volume_triangulation.begin_active();
         cell != volume_triangulation.end();
         cell++)
      {
        if (cell->at_boundary())
          {
            for (unsigned int f = 0; f < cell->n_faces(); f++)
              {
                if (cell->face(f)->at_boundary())
                  {
                    cell->face(f)->set_all_boundary_ids(cell->material_id());
                  }
              }
          }
      }

    /**
     * Extract the surface mesh from the volume mesh.
     *
     * N.B. Such extraction is only performed on the coarse volume mesh, no
     * matter whether this volume mesh has been refined. During the extraction,
     * the boundary ids instead of the cell material ids will be transferred
     * from the volume mesh to the surface mesh.
     */
    std::map<typename Triangulation<dim, spacedim>::cell_iterator,
             typename Triangulation<spacedim, spacedim>::face_iterator>
      map_from_surface_mesh_to_volume_mesh =
        GridGenerator::extract_boundary_mesh(volume_triangulation,
                                             surface_triangulation);

    /**
     * Check the material ids of the cells in the surface mesh.
     */
    for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
           surface_triangulation.begin_active();
         cell != surface_triangulation.end();
         cell++)
      {
        std::cout << "Material id for cell in surface mesh: "
                  << cell->material_id() << std::endl;
      }

    /**
     * N.B. When the volume mesh is refined, the extracted surface mesh will not
     * be refined.
     *
     * \alert{The MSH mesh file saved from deal.ii does not contain material id
     * information.}
     */
    volume_triangulation.refine_global(4);

    std::ofstream volume_mesh_file("volume-mesh-from-dealii.msh");
    write_msh_correct(volume_triangulation, volume_mesh_file);

    std::cout << "=== Volume mesh information ===" << std::endl;
    print_mesh_info(std::cout, volume_triangulation);

    for (typename Triangulation<spacedim>::active_cell_iterator cell =
           volume_triangulation.begin_active();
         cell != volume_triangulation.end();
         cell++)
      {
        std::cout << "Material id in volume mesh: " << cell->material_id()
                  << std::endl;
      }

    /**
     * Because the surface mesh is not refined with the volume mesh, we have to
     * define and associate manifold objects and perform surface mesh
     * refinement.
     *
     * N.B. The manifold objects defined for the surface mesh have different
     * dimensions than those for the volume mesh.
     */
    const SphericalManifold<dim, spacedim> left_ball_surface_manifold(
      Point<spacedim>(-1.5, 0, 0));
    const SphericalManifold<dim, spacedim> right_ball_surface_manifold(
      Point<spacedim>(1.5, 0, 0));

    for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
           surface_triangulation.begin_active();
         cell != surface_triangulation.end();
         cell++)
      {
        switch (cell->material_id())
          {
              case 0: {
                cell->set_all_manifold_ids(0);

                break;
              }
              case 1: {
                cell->set_all_manifold_ids(1);

                break;
              }
              default: {
                break;
              }
          }
      }

    surface_triangulation.set_manifold(0, left_ball_surface_manifold);
    surface_triangulation.set_manifold(1, right_ball_surface_manifold);

    surface_triangulation.refine_global(4);

    /**
     *  \alert{The MSH mesh file saved from deal.ii does not contain material id
     * information.}
     */
    std::ofstream surface_mesh_file("surface-mesh-from-dealii.msh");
    write_msh_correct(surface_triangulation, surface_mesh_file);

    std::cout << "=== Surface mesh information ===" << std::endl;
    print_mesh_info(std::cout, surface_triangulation);
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::set_dirichlet_boundary_ids(
    std::initializer_list<types::boundary_id> ilist)
  {
    boundary_ids_for_dirichlet_domain = ilist;
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::set_neumann_boundary_ids(
    std::initializer_list<types::boundary_id> ilist)
  {
    boundary_ids_for_neumann_domain = ilist;
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::generate_cell_iterators()
  {
    cell_iterators_for_dirichlet_space.reserve(
      dof_handler_for_dirichlet_space.get_triangulation().n_active_cells());
    for (const auto &cell :
         dof_handler_for_dirichlet_space.active_cell_iterators())
      {
        cell_iterators_for_dirichlet_space.push_back(cell);
      }

    cell_iterators_for_neumann_space.reserve(
      dof_handler_for_neumann_space.get_triangulation().n_active_cells());
    for (const auto &cell :
         dof_handler_for_neumann_space.active_cell_iterators())
      {
        cell_iterators_for_neumann_space.push_back(cell);
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::setup_system()
  {
    LogStream::Prefix prefix_string("setup_system");
#if ENABLE_NVTX == 1
    HierBEM::CUDAWrappers::NVTXRange nvtx_range("setup_system");
#endif

    Timer timer;
    timer.stop();

    switch (problem_type)
      {
          case DirichletBCProblem: {
            dof_handler_for_dirichlet_space.reinit(surface_triangulation);
            dof_handler_for_dirichlet_space.distribute_dofs(
              fe_for_dirichlet_space);
            dof_handler_for_neumann_space.reinit(surface_triangulation);
            dof_handler_for_neumann_space.distribute_dofs(fe_for_neumann_space);

            const unsigned int n_dofs_for_dirichlet_space =
              dof_handler_for_dirichlet_space.n_dofs();
            const unsigned int n_dofs_for_neumann_space =
              dof_handler_for_neumann_space.n_dofs();

            std::cout << "=== DoF information ===" << std::endl;
            std::cout << "Number of DoFs in Dirichlet space: "
                      << n_dofs_for_dirichlet_space << "\n";
            std::cout << "Number of DoFs in Neumann space: "
                      << n_dofs_for_neumann_space << std::endl;

            if (!is_use_hmat)
              {
                /**
                 * If full matrices are used for verification purpose,
                 * allocate memory for them here.
                 */
                V1_matrix.reinit(n_dofs_for_neumann_space,
                                 n_dofs_for_neumann_space);
                K2_matrix_with_mass_matrix.reinit(n_dofs_for_neumann_space,
                                                  n_dofs_for_dirichlet_space);
              }
            else
              {
                /**
                 * Build the DoF-to-cell topology.
                 */
                timer.start();

                generate_cell_iterators();
                build_dof_to_cell_topology(dof_to_cell_topo_for_dirichlet_space,
                                           cell_iterators_for_dirichlet_space,
                                           dof_handler_for_dirichlet_space);
                build_dof_to_cell_topology(dof_to_cell_topo_for_neumann_space,
                                           cell_iterators_for_neumann_space,
                                           dof_handler_for_neumann_space);

                timer.stop();
                print_wall_time(deallog, timer, "build dof-to-cell topology");

                /**
                 * Generate lists of DoF indices.
                 */
                dof_indices_for_dirichlet_space_on_dirichlet_domain.resize(
                  n_dofs_for_dirichlet_space);
                dof_indices_for_neumann_space_on_dirichlet_domain.resize(
                  n_dofs_for_neumann_space);

                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_dirichlet_space_on_dirichlet_domain);
                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_neumann_space_on_dirichlet_domain);

                /**
                 * Get the spatial coordinates of the support points.
                 */
                support_points_for_dirichlet_space_on_dirichlet_domain.resize(
                  n_dofs_for_dirichlet_space);
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_dirichlet_domain,
                  dof_handler_for_dirichlet_space,
                  support_points_for_dirichlet_space_on_dirichlet_domain);

                support_points_for_neumann_space_on_dirichlet_domain.resize(
                  n_dofs_for_neumann_space);
                // TODO In reality, different subdomains are assigned different
                // mapping objects, which should be handled separately.
                // Furthermore, the Jacobian of the mapping should also be taken
                // into account in numerical quadrature.
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_dirichlet_domain,
                  dof_handler_for_neumann_space,
                  support_points_for_neumann_space_on_dirichlet_domain);

                /**
                 * Calculate the average mesh cell size at each support point.
                 */
                dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain
                  .assign(n_dofs_for_dirichlet_space, 0);
                DoFToolsExt::map_dofs_to_average_cell_size(
                  dof_handler_for_dirichlet_space,
                  dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);

                dof_average_cell_size_for_neumann_space_on_dirichlet_domain
                  .assign(n_dofs_for_neumann_space, 0);
                DoFToolsExt::map_dofs_to_average_cell_size(
                  dof_handler_for_neumann_space,
                  dof_average_cell_size_for_neumann_space_on_dirichlet_domain);

                /**
                 * Initialize the cluster trees.
                 */
                timer.start();

                ct_for_dirichlet_space_on_dirichlet_domain = ClusterTree<
                  spacedim>(
                  dof_indices_for_dirichlet_space_on_dirichlet_domain,
                  support_points_for_dirichlet_space_on_dirichlet_domain,
                  dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain,
                  n_min_for_ct);
                ct_for_neumann_space_on_dirichlet_domain =
                  ClusterTree<spacedim>(
                    dof_indices_for_neumann_space_on_dirichlet_domain,
                    support_points_for_neumann_space_on_dirichlet_domain,
                    dof_average_cell_size_for_neumann_space_on_dirichlet_domain,
                    n_min_for_ct);

                /**
                 * Partition the cluster trees.
                 */
                ct_for_dirichlet_space_on_dirichlet_domain.partition(
                  support_points_for_dirichlet_space_on_dirichlet_domain,
                  dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);
                ct_for_neumann_space_on_dirichlet_domain.partition(
                  support_points_for_neumann_space_on_dirichlet_domain,
                  dof_average_cell_size_for_neumann_space_on_dirichlet_domain);

#if ENABLE_DEBUG == 1 && MESSAGE_LEVEL >= 3
                /**
                 * @internal Visualize partitioned mesh on different levels
                 * (from level #1 to the maximum level) in the cluster trees.
                 */
                std::vector<typename ClusterTree<spacedim>::node_pointer_type>
                  cluster_set;
                Vector<float>
                  dof_cluster_indices_for_dirichlet_space_on_dirichlet_domain(
                    n_dofs_for_dirichlet_space);
                Vector<float>
                  dof_cluster_indices_for_neumann_space_on_dirichlet_domain(
                    n_dofs_for_neumann_space);

                std::ofstream
                  cluster_level_vtk_for_dirichlet_space_on_dirichlet_domain;
                std::ofstream
                  cluster_level_vtk_for_neumann_space_on_dirichlet_domain;

                DataOut<dim, spacedim>
                  cluster_level_data_out_for_dirichlet_space_on_dirichlet_domain;
                DataOut<dim, spacedim>
                  cluster_level_data_out_for_neumann_space_on_dirichlet_domain;

                cluster_level_vtk_for_dirichlet_space_on_dirichlet_domain.open(
                  "cluster_levels_for_dirichlet_space_on_dirichlet_domain.vtk",
                  std::ofstream::out);
                cluster_level_vtk_for_neumann_space_on_dirichlet_domain.open(
                  "cluster_levels_for_neumann_space_on_dirichlet_domain.vtk",
                  std::ofstream::out);

                for (unsigned int l = 1;
                     l <=
                     ct_for_dirichlet_space_on_dirichlet_domain.get_max_level();
                     l++)
                  {
                    ct_for_dirichlet_space_on_dirichlet_domain
                      .build_cluster_set_at_level(l, cluster_set);
                    /**
                     * @internal Assign the index for each cluster in the
                     * cluster set to its contained DoFs.
                     */
                    for (unsigned int c = 0; c < cluster_set.size(); c++)
                      {
                        const std::array<types::global_dof_index,
                                         2> &cluster_index_range =
                          cluster_set[c]->get_data_pointer()->get_index_range();

                        for (types::global_dof_index dof_index =
                               cluster_index_range[0];
                             dof_index < cluster_index_range[1];
                             dof_index++)
                          {
                            dof_cluster_indices_for_dirichlet_space_on_dirichlet_domain(
                              ct_for_dirichlet_space_on_dirichlet_domain
                                .get_internal_to_external_dof_numbering()
                                  [dof_index]) = c;
                          }
                      }

                    cluster_level_data_out_for_dirichlet_space_on_dirichlet_domain
                      .add_data_vector(
                        dof_handler_for_dirichlet_space,
                        dof_cluster_indices_for_dirichlet_space_on_dirichlet_domain,
                        std::string(
                          "cluster_for_dirichlet_space_on_dirichlet_domain_on_level_#") +
                          std::to_string(l));
                  }

                for (unsigned int l = 1;
                     l <=
                     ct_for_neumann_space_on_dirichlet_domain.get_max_level();
                     l++)
                  {
                    ct_for_neumann_space_on_dirichlet_domain
                      .build_cluster_set_at_level(l, cluster_set);
                    /**
                     * @internal Assign the index for each cluster in the
                     * cluster set to its contained DoFs.
                     */
                    for (unsigned int c = 0; c < cluster_set.size(); c++)
                      {
                        const std::array<types::global_dof_index,
                                         2> &cluster_index_range =
                          cluster_set[c]->get_data_pointer()->get_index_range();

                        for (types::global_dof_index dof_index =
                               cluster_index_range[0];
                             dof_index < cluster_index_range[1];
                             dof_index++)
                          {
                            dof_cluster_indices_for_neumann_space_on_dirichlet_domain(
                              ct_for_neumann_space_on_dirichlet_domain
                                .get_internal_to_external_dof_numbering()
                                  [dof_index]) = c;
                          }
                      }

                    cluster_level_data_out_for_neumann_space_on_dirichlet_domain
                      .add_data_vector(
                        dof_handler_for_neumann_space,
                        dof_cluster_indices_for_neumann_space_on_dirichlet_domain,
                        std::string(
                          "cluster_for_neumann_space_on_dirichlet_domain_on_level_#") +
                          std::to_string(l));
                  }

                cluster_level_data_out_for_dirichlet_space_on_dirichlet_domain
                  .build_patches();
                cluster_level_data_out_for_neumann_space_on_dirichlet_domain
                  .build_patches();
                cluster_level_data_out_for_dirichlet_space_on_dirichlet_domain
                  .write_vtk(
                    cluster_level_vtk_for_dirichlet_space_on_dirichlet_domain);
                cluster_level_data_out_for_neumann_space_on_dirichlet_domain
                  .write_vtk(
                    cluster_level_vtk_for_neumann_space_on_dirichlet_domain);

                cluster_level_vtk_for_dirichlet_space_on_dirichlet_domain
                  .close();
                cluster_level_vtk_for_neumann_space_on_dirichlet_domain.close();
#endif

                /**
                 * Get the external-to-internal and internal-to-external DoF
                 * numberings.
                 */
                dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain =
                  &(ct_for_dirichlet_space_on_dirichlet_domain
                      .get_external_to_internal_dof_numbering());
                dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain =
                  &(ct_for_dirichlet_space_on_dirichlet_domain
                      .get_internal_to_external_dof_numbering());
                dof_e2i_numbering_for_neumann_space_on_dirichlet_domain =
                  &(ct_for_neumann_space_on_dirichlet_domain
                      .get_external_to_internal_dof_numbering());
                dof_i2e_numbering_for_neumann_space_on_dirichlet_domain =
                  &(ct_for_neumann_space_on_dirichlet_domain
                      .get_internal_to_external_dof_numbering());

                timer.stop();
                print_wall_time(deallog, timer, "build cluster trees");

#if ENABLE_DEBUG == 1 && MESSAGE_LEVEL >= 3
                {
                  /**
                   * @internal Generate the graph for the cluster tree in the
                   * PlantUML format.
                   */
                  std::ofstream ct_dirichlet_space_out(
                    "ct-dirichlet-after-dof-numbering-reorder.puml");
                  std::ofstream ct_neumann_space_out(
                    "ct-neumann-after-dof-numbering-reorder.puml");

                  ct_for_dirichlet_space_on_dirichlet_domain
                    .print_tree_info_as_dot(ct_dirichlet_space_out);
                  ct_for_neumann_space_on_dirichlet_domain
                    .print_tree_info_as_dot(ct_neumann_space_out);

                  ct_dirichlet_space_out.close();
                  ct_neumann_space_out.close();
                }
#endif

                /**
                 * Create the block cluster trees.
                 */
                timer.start();

                bct_for_bilinear_form_V1 = BlockClusterTree<spacedim>(
                  ct_for_neumann_space_on_dirichlet_domain,
                  ct_for_neumann_space_on_dirichlet_domain,
                  eta,
                  n_min_for_bct);
                bct_for_bilinear_form_K2 = BlockClusterTree<spacedim>(
                  ct_for_neumann_space_on_dirichlet_domain,
                  ct_for_dirichlet_space_on_dirichlet_domain,
                  eta,
                  n_min_for_bct);

                /**
                 * Partition the block cluster trees.
                 */
                bct_for_bilinear_form_V1.partition(
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  support_points_for_neumann_space_on_dirichlet_domain,
                  dof_average_cell_size_for_neumann_space_on_dirichlet_domain);
                bct_for_bilinear_form_K2.partition(
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  support_points_for_neumann_space_on_dirichlet_domain,
                  support_points_for_dirichlet_space_on_dirichlet_domain,
                  dof_average_cell_size_for_neumann_space_on_dirichlet_domain,
                  dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);

                timer.stop();
                print_wall_time(deallog, timer, "build block cluster trees");

                /**
                 * Initialize \hmatrices.
                 */
                timer.start();

                V1_hmat = HMatrixSymm<spacedim>(bct_for_bilinear_form_V1,
                                                max_hmat_rank);

                std::cout << "=== Leaf set information of V1_hmat ==="
                          << std::endl;
                V1_hmat.print_leaf_set_info(std::cout);

                K2_hmat_with_mass_matrix =
                  HMatrix<spacedim>(bct_for_bilinear_form_K2,
                                    max_hmat_rank,
                                    HMatrixSupport::Property::general,
                                    HMatrixSupport::BlockType::diagonal_block);

                std::cout
                  << "=== Leaf set information of K2_hmat_with_mass_matrix ==="
                  << std::endl;
                K2_hmat_with_mass_matrix.print_leaf_set_info(std::cout);

                timer.stop();
                print_wall_time(deallog, timer, "initialize H-matrices");
              }

            /**
             * Interpolate the Dirichlet boundary data.
             *
             * TODO When there are multiple Dirichlet subdomains assigned
             * different boundary values, they should be handled separately.
             */
            timer.start();

            dirichlet_bc.reinit(n_dofs_for_dirichlet_space);
            VectorTools::interpolate(dof_handler_for_dirichlet_space,
                                     *dirichlet_bc_functor_ptr,
                                     dirichlet_bc);

            if (is_use_hmat)
              {
                /**
                 * Permute the Dirichlet boundary data by following the mapping
                 * from internal to external DoF numbering.
                 */
                dirichlet_bc_internal_dof_numbering.reinit(
                  n_dofs_for_dirichlet_space);
                permute_vector(
                  dirichlet_bc,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  dirichlet_bc_internal_dof_numbering);
              }

            timer.stop();
            print_wall_time(deallog, timer, "interpolate boundary condition");

            /**
             * Allocate memory for the right-hand-side vector and solution
             * vector.
             */
            system_rhs_on_dirichlet_domain.reinit(n_dofs_for_neumann_space);
            neumann_data.reinit(n_dofs_for_neumann_space);

            if (is_use_hmat)
              {
                // Solution vector in the internal numbering.
                neumann_data_on_dirichlet_domain_internal_dof_numbering.reinit(
                  n_dofs_for_neumann_space);
              }

            break;
          }
          case NeumannBCProblem: {
            dof_handler_for_dirichlet_space.reinit(surface_triangulation);
            dof_handler_for_dirichlet_space.distribute_dofs(
              fe_for_dirichlet_space);
            dof_handler_for_neumann_space.reinit(surface_triangulation);
            dof_handler_for_neumann_space.distribute_dofs(fe_for_neumann_space);

            const unsigned int n_dofs_for_dirichlet_space =
              dof_handler_for_dirichlet_space.n_dofs();
            const unsigned int n_dofs_for_neumann_space =
              dof_handler_for_neumann_space.n_dofs();

            std::cout << "=== DoF information ===" << std::endl;
            std::cout << "Number of DoFs in Dirichlet space: "
                      << n_dofs_for_dirichlet_space << "\n";
            std::cout << "Number of DoFs in Neumann space: "
                      << n_dofs_for_neumann_space << std::endl;

            if (!is_use_hmat)
              {
                /**
                 * If full matrices are used for verification purpose,
                 * allocate memory for them here.
                 */
                D1_matrix.reinit(n_dofs_for_dirichlet_space,
                                 n_dofs_for_dirichlet_space);
                K_prime2_matrix_with_mass_matrix.reinit(
                  n_dofs_for_dirichlet_space, n_dofs_for_neumann_space);

                /**
                 * SLP matrix for solving the natural density \f$w_{\rm eq}\f$.
                 */
                V1_matrix.reinit(n_dofs_for_neumann_space,
                                 n_dofs_for_neumann_space);
              }
            else
              {
                /**
                 * Build the DoF-to-cell topology.
                 */
                timer.start();

                generate_cell_iterators();
                build_dof_to_cell_topology(dof_to_cell_topo_for_dirichlet_space,
                                           cell_iterators_for_dirichlet_space,
                                           dof_handler_for_dirichlet_space);
                build_dof_to_cell_topology(dof_to_cell_topo_for_neumann_space,
                                           cell_iterators_for_neumann_space,
                                           dof_handler_for_neumann_space);

                timer.stop();
                print_wall_time(deallog, timer, "build dof-to-cell topology");

                /**
                 * Generate lists of DoF indices.
                 */
                dof_indices_for_dirichlet_space_on_neumann_domain.resize(
                  n_dofs_for_dirichlet_space);
                dof_indices_for_neumann_space_on_neumann_domain.resize(
                  n_dofs_for_neumann_space);

                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_dirichlet_space_on_neumann_domain);
                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_neumann_space_on_neumann_domain);

                /**
                 * Get the spatial coordinates of the support points.
                 */
                support_points_for_dirichlet_space_on_neumann_domain.resize(
                  n_dofs_for_dirichlet_space);
                // TODO In reality, different subdomains are assigned different
                // mapping objects, which should be handled separately.
                // Furthermore, the Jacobian of the mapping should also be taken
                // into account in numerical quadrature.
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_neumann_domain,
                  dof_handler_for_dirichlet_space,
                  support_points_for_dirichlet_space_on_neumann_domain);

                support_points_for_neumann_space_on_neumann_domain.resize(
                  n_dofs_for_neumann_space);
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_neumann_domain,
                  dof_handler_for_neumann_space,
                  support_points_for_neumann_space_on_neumann_domain);

                /**
                 * Calculate the average mesh cell size at each support point.
                 */
                dof_average_cell_size_for_dirichlet_space_on_neumann_domain
                  .assign(n_dofs_for_dirichlet_space, 0);
                DoFToolsExt::map_dofs_to_average_cell_size(
                  dof_handler_for_dirichlet_space,
                  dof_average_cell_size_for_dirichlet_space_on_neumann_domain);

                dof_average_cell_size_for_neumann_space_on_neumann_domain
                  .assign(n_dofs_for_neumann_space, 0);
                DoFToolsExt::map_dofs_to_average_cell_size(
                  dof_handler_for_neumann_space,
                  dof_average_cell_size_for_neumann_space_on_neumann_domain);

                /**
                 * Initialize the cluster trees.
                 */
                timer.start();

                ct_for_dirichlet_space_on_neumann_domain =
                  ClusterTree<spacedim>(
                    dof_indices_for_dirichlet_space_on_neumann_domain,
                    support_points_for_dirichlet_space_on_neumann_domain,
                    dof_average_cell_size_for_dirichlet_space_on_neumann_domain,
                    n_min_for_ct);
                ct_for_neumann_space_on_neumann_domain = ClusterTree<spacedim>(
                  dof_indices_for_neumann_space_on_neumann_domain,
                  support_points_for_neumann_space_on_neumann_domain,
                  dof_average_cell_size_for_neumann_space_on_neumann_domain,
                  n_min_for_ct);

                /**
                 * Partition the cluster trees.
                 */
                ct_for_dirichlet_space_on_neumann_domain.partition(
                  support_points_for_dirichlet_space_on_neumann_domain,
                  dof_average_cell_size_for_dirichlet_space_on_neumann_domain);
                ct_for_neumann_space_on_neumann_domain.partition(
                  support_points_for_neumann_space_on_neumann_domain,
                  dof_average_cell_size_for_neumann_space_on_neumann_domain);

                /**
                 * Get the external-to-internal and internal-to-external DoF
                 * numberings.
                 */
                dof_e2i_numbering_for_dirichlet_space_on_neumann_domain =
                  &(ct_for_dirichlet_space_on_neumann_domain
                      .get_external_to_internal_dof_numbering());
                dof_i2e_numbering_for_dirichlet_space_on_neumann_domain =
                  &(ct_for_dirichlet_space_on_neumann_domain
                      .get_internal_to_external_dof_numbering());
                dof_e2i_numbering_for_neumann_space_on_neumann_domain =
                  &(ct_for_neumann_space_on_neumann_domain
                      .get_external_to_internal_dof_numbering());
                dof_i2e_numbering_for_neumann_space_on_neumann_domain =
                  &(ct_for_neumann_space_on_neumann_domain
                      .get_internal_to_external_dof_numbering());

                timer.stop();
                print_wall_time(deallog, timer, "build cluster trees");

                /**
                 * Create the block cluster trees.
                 */
                timer.start();

                bct_for_bilinear_form_D1 = BlockClusterTree<spacedim>(
                  ct_for_dirichlet_space_on_neumann_domain,
                  ct_for_dirichlet_space_on_neumann_domain,
                  eta,
                  n_min_for_bct);
                bct_for_bilinear_form_K_prime2 = BlockClusterTree<spacedim>(
                  ct_for_dirichlet_space_on_neumann_domain,
                  ct_for_neumann_space_on_neumann_domain,
                  eta,
                  n_min_for_bct);
                bct_for_bilinear_form_V1 = BlockClusterTree<spacedim>(
                  ct_for_neumann_space_on_neumann_domain,
                  ct_for_neumann_space_on_neumann_domain,
                  eta,
                  n_min_for_bct);

                /**
                 * Partition the block cluster trees.
                 */
                bct_for_bilinear_form_D1.partition(
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  support_points_for_dirichlet_space_on_neumann_domain,
                  dof_average_cell_size_for_dirichlet_space_on_neumann_domain);
                bct_for_bilinear_form_K_prime2.partition(
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  support_points_for_dirichlet_space_on_neumann_domain,
                  support_points_for_neumann_space_on_neumann_domain,
                  dof_average_cell_size_for_dirichlet_space_on_neumann_domain,
                  dof_average_cell_size_for_neumann_space_on_neumann_domain);
                bct_for_bilinear_form_V1.partition(
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  support_points_for_neumann_space_on_neumann_domain,
                  dof_average_cell_size_for_neumann_space_on_neumann_domain);

                timer.stop();
                print_wall_time(deallog, timer, "build block cluster trees");

                /**
                 * Initialize \hmatrices.
                 */
                timer.start();

                D1_hmat = HMatrixSymm<spacedim>(bct_for_bilinear_form_D1,
                                                max_hmat_rank);

                std::cout << "=== Leaf set information of D1_hmat ==="
                          << std::endl;
                D1_hmat.print_leaf_set_info(std::cout);

                K_prime2_hmat_with_mass_matrix =
                  HMatrix<spacedim>(bct_for_bilinear_form_K_prime2,
                                    max_hmat_rank,
                                    HMatrixSupport::Property::general,
                                    HMatrixSupport::BlockType::diagonal_block);

                std::cout
                  << "=== Leaf set information of K_prime2_hmat_with_mass_matrix ==="
                  << std::endl;
                K_prime2_hmat_with_mass_matrix.print_leaf_set_info(std::cout);

                /**
                 * SLP matrix for solving the natural density \f$w_{\rm eq}\f$.
                 */
                V1_hmat = HMatrixSymm<spacedim>(bct_for_bilinear_form_V1,
                                                max_hmat_rank);

                std::cout << "=== Leaf set information of V1_hmat ==="
                          << std::endl;
                V1_hmat.print_leaf_set_info(std::cout);

                timer.stop();
                print_wall_time(deallog, timer, "initialize H-matrices");
              }

            /**
             * Interpolate the Neumann boundary data.
             *
             * TODO When there are multiple Neumann subdomains assigned
             * different boundary values, they should be handled separately.
             */
            timer.start();

            neumann_bc.reinit(n_dofs_for_neumann_space);
            VectorTools::interpolate(dof_handler_for_neumann_space,
                                     *neumann_bc_functor_ptr,
                                     neumann_bc);

            if (is_use_hmat)
              {
                /**
                 * Permute the Neumann boundary data by following the mapping
                 * from internal to external DoF numbering.
                 */
                neumann_bc_internal_dof_numbering.reinit(
                  n_dofs_for_neumann_space);
                permute_vector(
                  neumann_bc,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  neumann_bc_internal_dof_numbering);
              }

            timer.stop();
            print_wall_time(deallog, timer, "interpolate boundary condition");

            /**
             * Allocate memory for the natural density \f$w_{\rm eq}\in
             * H^{-1/2}(\Gamma)\f$ and its associated right hand side vector.
             */
            natural_density.reinit(n_dofs_for_neumann_space);
            system_rhs_for_natural_density.reinit(n_dofs_for_neumann_space);

            /**
             * Allocate memory for the product of mass matrix and the natural
             * density.
             */
            mass_vmult_weq.reinit(n_dofs_for_dirichlet_space);

            /**
             * Allocate memory for the right-hand-side vector and solution
             * vector.
             */
            system_rhs_on_neumann_domain.reinit(n_dofs_for_dirichlet_space);
            dirichlet_data.reinit(n_dofs_for_dirichlet_space);

            if (is_use_hmat)
              {
                // Solution vector in the internal numbering
                dirichlet_data_on_neumann_domain_internal_dof_numbering.reinit(
                  n_dofs_for_dirichlet_space);
              }

            break;
          }
          case MixedBCProblem: {
            Assert(is_use_hmat, ExcInternalError());

            // Initialize DoF handlers.
            dof_handler_for_dirichlet_space.reinit(surface_triangulation);
            dof_handler_for_dirichlet_space.distribute_dofs(
              fe_for_dirichlet_space);
            dof_handler_for_neumann_space.reinit(surface_triangulation);
            dof_handler_for_neumann_space.distribute_dofs(fe_for_neumann_space);

            // Generate DoF selectors for the Dirichlet space on the extended
            // Dirichlet domain and retracted Neumann domain.
            timer.start();

            dof_selectors_for_dirichlet_space_on_dirichlet_domain.resize(
              dof_handler_for_dirichlet_space.n_dofs());
            dof_selectors_for_dirichlet_space_on_neumann_domain.resize(
              dof_handler_for_dirichlet_space.n_dofs());
            DoFToolsExt::extract_material_domain_dofs(
              dof_handler_for_dirichlet_space,
              boundary_ids_for_dirichlet_domain,
              dof_selectors_for_dirichlet_space_on_dirichlet_domain);

            local_to_full_dirichlet_dof_indices_on_dirichlet_domain.reserve(
              dof_handler_for_dirichlet_space.n_dofs());
            local_to_full_dirichlet_dof_indices_on_neumann_domain.reserve(
              dof_handler_for_dirichlet_space.n_dofs());

            for (types::global_dof_index i = 0;
                 i < dof_handler_for_dirichlet_space.n_dofs();
                 i++)
              {
                if (dof_selectors_for_dirichlet_space_on_dirichlet_domain[i])
                  {
                    dof_selectors_for_dirichlet_space_on_neumann_domain[i] =
                      false;
                    local_to_full_dirichlet_dof_indices_on_dirichlet_domain
                      .push_back(i);
                  }
                else
                  {
                    dof_selectors_for_dirichlet_space_on_neumann_domain[i] =
                      true;
                    local_to_full_dirichlet_dof_indices_on_neumann_domain
                      .push_back(i);
                  }
              }

            // Generate DoF selectors for the Neumann space on the Dirichlet
            // domain and Neumann domain.
            dof_selectors_for_neumann_space_on_dirichlet_domain.resize(
              dof_handler_for_neumann_space.n_dofs());
            dof_selectors_for_neumann_space_on_neumann_domain.resize(
              dof_handler_for_neumann_space.n_dofs());
            DoFToolsExt::extract_material_domain_dofs(
              dof_handler_for_neumann_space,
              boundary_ids_for_dirichlet_domain,
              dof_selectors_for_neumann_space_on_dirichlet_domain);

            local_to_full_neumann_dof_indices_on_dirichlet_domain.reserve(
              dof_handler_for_neumann_space.n_dofs());
            local_to_full_neumann_dof_indices_on_neumann_domain.reserve(
              dof_handler_for_neumann_space.n_dofs());

            for (types::global_dof_index i = 0;
                 i < dof_handler_for_neumann_space.n_dofs();
                 i++)
              {
                if (dof_selectors_for_neumann_space_on_dirichlet_domain[i])
                  {
                    dof_selectors_for_neumann_space_on_neumann_domain[i] =
                      false;
                    local_to_full_neumann_dof_indices_on_dirichlet_domain
                      .push_back(i);
                  }
                else
                  {
                    dof_selectors_for_neumann_space_on_neumann_domain[i] = true;
                    local_to_full_neumann_dof_indices_on_neumann_domain
                      .push_back(i);
                  }
              }

            timer.stop();
            print_wall_time(deallog, timer, "generate DoF selectors");

            // Get the number of effective DoF number for each DoF handler.
            const unsigned int n_dofs_for_dirichlet_space_on_dirichlet_domain =
              local_to_full_dirichlet_dof_indices_on_dirichlet_domain.size();
            const unsigned int n_dofs_for_dirichlet_space_on_neumann_domain =
              local_to_full_dirichlet_dof_indices_on_neumann_domain.size();
            const unsigned int n_dofs_for_neumann_space_on_dirichlet_domain =
              local_to_full_neumann_dof_indices_on_dirichlet_domain.size();
            const unsigned int n_dofs_for_neumann_space_on_neumann_domain =
              local_to_full_neumann_dof_indices_on_neumann_domain.size();

            std::cout << "=== DoF information ===" << std::endl;
            std::cout
              << "Number of DoFs in Dirichlet space on Dirichlet domain: "
              << n_dofs_for_dirichlet_space_on_dirichlet_domain << "\n";
            std::cout << "Number of DoFs in Dirichlet space on Neumann domain: "
                      << n_dofs_for_dirichlet_space_on_neumann_domain << "\n";
            std::cout << "Number of DoFs in Neumann space on Dirichlet domain: "
                      << n_dofs_for_neumann_space_on_dirichlet_domain << "\n";
            std::cout << "Number of DoFs in Neumann space on Neumann domain: "
                      << n_dofs_for_neumann_space_on_neumann_domain
                      << std::endl;

            /**
             * Build the DoF-to-cell topology.
             *
             * \mynote{Access of this topology for the Dirichlet space
             * requires the map from local to full DoF indices.}
             */
            timer.start();

            generate_cell_iterators();
            build_dof_to_cell_topology(dof_to_cell_topo_for_dirichlet_space,
                                       cell_iterators_for_dirichlet_space,
                                       dof_handler_for_dirichlet_space);
            build_dof_to_cell_topology(dof_to_cell_topo_for_neumann_space,
                                       cell_iterators_for_neumann_space,
                                       dof_handler_for_neumann_space);

            timer.stop();
            print_wall_time(deallog, timer, "build dof-to-cell topology");

            /**
             * Generate lists of DoF indices.
             *
             * \mynote{N.B. For the Dirichlet space, some DoFs have been
             * excluded.}
             */
            dof_indices_for_dirichlet_space_on_dirichlet_domain.resize(
              n_dofs_for_dirichlet_space_on_dirichlet_domain);
            dof_indices_for_dirichlet_space_on_neumann_domain.resize(
              n_dofs_for_dirichlet_space_on_neumann_domain);
            dof_indices_for_neumann_space_on_dirichlet_domain.resize(
              n_dofs_for_neumann_space_on_dirichlet_domain);
            dof_indices_for_neumann_space_on_neumann_domain.resize(
              n_dofs_for_neumann_space_on_neumann_domain);

            gen_linear_indices<vector_uta, types::global_dof_index>(
              dof_indices_for_dirichlet_space_on_dirichlet_domain);
            gen_linear_indices<vector_uta, types::global_dof_index>(
              dof_indices_for_dirichlet_space_on_neumann_domain);
            gen_linear_indices<vector_uta, types::global_dof_index>(
              dof_indices_for_neumann_space_on_dirichlet_domain);
            gen_linear_indices<vector_uta, types::global_dof_index>(
              dof_indices_for_neumann_space_on_neumann_domain);

            /**
             * Get the spatial coordinates of the support points.
             */
            support_points_for_dirichlet_space_on_dirichlet_domain.resize(
              n_dofs_for_dirichlet_space_on_dirichlet_domain);
            DoFToolsExt::map_dofs_to_support_points(
              kx_mapping_for_dirichlet_domain,
              dof_handler_for_dirichlet_space,
              local_to_full_dirichlet_dof_indices_on_dirichlet_domain,
              support_points_for_dirichlet_space_on_dirichlet_domain);

            support_points_for_dirichlet_space_on_neumann_domain.resize(
              n_dofs_for_dirichlet_space_on_neumann_domain);
            DoFToolsExt::map_dofs_to_support_points(
              kx_mapping_for_neumann_domain,
              dof_handler_for_dirichlet_space,
              local_to_full_dirichlet_dof_indices_on_neumann_domain,
              support_points_for_dirichlet_space_on_neumann_domain);

            support_points_for_neumann_space_on_dirichlet_domain.resize(
              n_dofs_for_neumann_space_on_dirichlet_domain);
            DoFToolsExt::map_dofs_to_support_points(
              kx_mapping_for_dirichlet_domain,
              dof_handler_for_neumann_space,
              local_to_full_neumann_dof_indices_on_dirichlet_domain,
              support_points_for_neumann_space_on_dirichlet_domain);

            support_points_for_neumann_space_on_neumann_domain.resize(
              n_dofs_for_neumann_space_on_neumann_domain);
            DoFToolsExt::map_dofs_to_support_points(
              kx_mapping_for_neumann_domain,
              dof_handler_for_neumann_space,
              local_to_full_neumann_dof_indices_on_neumann_domain,
              support_points_for_neumann_space_on_neumann_domain);

            /**
             * Calculate the average mesh cell size at each support point.
             */
            dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain
              .assign(n_dofs_for_dirichlet_space_on_dirichlet_domain, 0);
            DoFToolsExt::map_dofs_to_average_cell_size(
              dof_handler_for_dirichlet_space,
              local_to_full_dirichlet_dof_indices_on_dirichlet_domain,
              dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);

            dof_average_cell_size_for_dirichlet_space_on_neumann_domain.assign(
              n_dofs_for_dirichlet_space_on_neumann_domain, 0);
            DoFToolsExt::map_dofs_to_average_cell_size(
              dof_handler_for_dirichlet_space,
              local_to_full_dirichlet_dof_indices_on_neumann_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain);

            dof_average_cell_size_for_neumann_space_on_dirichlet_domain.assign(
              n_dofs_for_neumann_space_on_dirichlet_domain, 0);
            DoFToolsExt::map_dofs_to_average_cell_size(
              dof_handler_for_neumann_space,
              local_to_full_neumann_dof_indices_on_dirichlet_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain);

            dof_average_cell_size_for_neumann_space_on_neumann_domain.assign(
              n_dofs_for_neumann_space_on_neumann_domain, 0);
            DoFToolsExt::map_dofs_to_average_cell_size(
              dof_handler_for_neumann_space,
              local_to_full_neumann_dof_indices_on_neumann_domain,
              dof_average_cell_size_for_neumann_space_on_neumann_domain);

            /**
             * Initialize the cluster trees.
             */
            timer.start();

            ct_for_dirichlet_space_on_dirichlet_domain = ClusterTree<spacedim>(
              dof_indices_for_dirichlet_space_on_dirichlet_domain,
              support_points_for_dirichlet_space_on_dirichlet_domain,
              dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain,
              n_min_for_ct);
            ct_for_dirichlet_space_on_neumann_domain = ClusterTree<spacedim>(
              dof_indices_for_dirichlet_space_on_neumann_domain,
              support_points_for_dirichlet_space_on_neumann_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain,
              n_min_for_ct);
            ct_for_neumann_space_on_dirichlet_domain = ClusterTree<spacedim>(
              dof_indices_for_neumann_space_on_dirichlet_domain,
              support_points_for_neumann_space_on_dirichlet_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain,
              n_min_for_ct);
            ct_for_neumann_space_on_neumann_domain = ClusterTree<spacedim>(
              dof_indices_for_neumann_space_on_neumann_domain,
              support_points_for_neumann_space_on_neumann_domain,
              dof_average_cell_size_for_neumann_space_on_neumann_domain,
              n_min_for_ct);

            /**
             * Partition the cluster trees.
             */
            ct_for_dirichlet_space_on_dirichlet_domain.partition(
              support_points_for_dirichlet_space_on_dirichlet_domain,
              dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);
            ct_for_dirichlet_space_on_neumann_domain.partition(
              support_points_for_dirichlet_space_on_neumann_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain);
            ct_for_neumann_space_on_dirichlet_domain.partition(
              support_points_for_neumann_space_on_dirichlet_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain);
            ct_for_neumann_space_on_neumann_domain.partition(
              support_points_for_neumann_space_on_neumann_domain,
              dof_average_cell_size_for_neumann_space_on_neumann_domain);

            /**
             * Get the external-to-internal and internal-to-external DoF
             * numberings.
             */
            dof_e2i_numbering_for_dirichlet_space_on_dirichlet_domain =
              &(ct_for_dirichlet_space_on_dirichlet_domain
                  .get_external_to_internal_dof_numbering());
            dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain =
              &(ct_for_dirichlet_space_on_dirichlet_domain
                  .get_internal_to_external_dof_numbering());
            dof_e2i_numbering_for_dirichlet_space_on_neumann_domain =
              &(ct_for_dirichlet_space_on_neumann_domain
                  .get_external_to_internal_dof_numbering());
            dof_i2e_numbering_for_dirichlet_space_on_neumann_domain =
              &(ct_for_dirichlet_space_on_neumann_domain
                  .get_internal_to_external_dof_numbering());
            dof_e2i_numbering_for_neumann_space_on_dirichlet_domain =
              &(ct_for_neumann_space_on_dirichlet_domain
                  .get_external_to_internal_dof_numbering());
            dof_i2e_numbering_for_neumann_space_on_dirichlet_domain =
              &(ct_for_neumann_space_on_dirichlet_domain
                  .get_internal_to_external_dof_numbering());
            dof_e2i_numbering_for_neumann_space_on_neumann_domain =
              &(ct_for_neumann_space_on_neumann_domain
                  .get_external_to_internal_dof_numbering());
            dof_i2e_numbering_for_neumann_space_on_neumann_domain =
              &(ct_for_neumann_space_on_neumann_domain
                  .get_internal_to_external_dof_numbering());

            timer.stop();
            print_wall_time(deallog, timer, "build cluster trees");

            /**
             * Create the block cluster trees.
             */
            timer.start();

            bct_for_bilinear_form_V1 = BlockClusterTree<spacedim>(
              ct_for_neumann_space_on_dirichlet_domain,
              ct_for_neumann_space_on_dirichlet_domain,
              eta,
              n_min_for_bct);
            bct_for_bilinear_form_K1 = BlockClusterTree<spacedim>(
              ct_for_neumann_space_on_dirichlet_domain,
              ct_for_dirichlet_space_on_neumann_domain,
              eta,
              n_min_for_bct);
            bct_for_bilinear_form_D1 = BlockClusterTree<spacedim>(
              ct_for_dirichlet_space_on_neumann_domain,
              ct_for_dirichlet_space_on_neumann_domain,
              eta,
              n_min_for_bct);
            bct_for_bilinear_form_K2 = BlockClusterTree<spacedim>(
              ct_for_neumann_space_on_dirichlet_domain,
              ct_for_dirichlet_space_on_dirichlet_domain,
              eta,
              n_min_for_bct);
            bct_for_bilinear_form_V2 = BlockClusterTree<spacedim>(
              ct_for_neumann_space_on_dirichlet_domain,
              ct_for_neumann_space_on_neumann_domain,
              eta,
              n_min_for_bct);
            bct_for_bilinear_form_D2 = BlockClusterTree<spacedim>(
              ct_for_dirichlet_space_on_neumann_domain,
              ct_for_dirichlet_space_on_dirichlet_domain,
              eta,
              n_min_for_bct);
            bct_for_bilinear_form_K_prime2 = BlockClusterTree<spacedim>(
              ct_for_dirichlet_space_on_neumann_domain,
              ct_for_neumann_space_on_neumann_domain,
              eta,
              n_min_for_bct);

            /**
             * Partition the block cluster trees.
             */
            bct_for_bilinear_form_V1.partition(
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              support_points_for_neumann_space_on_dirichlet_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain);
            bct_for_bilinear_form_K1.partition(
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              support_points_for_neumann_space_on_dirichlet_domain,
              support_points_for_dirichlet_space_on_neumann_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain);
            bct_for_bilinear_form_D1.partition(
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              support_points_for_dirichlet_space_on_neumann_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain);
            bct_for_bilinear_form_K2.partition(
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
              support_points_for_neumann_space_on_dirichlet_domain,
              support_points_for_dirichlet_space_on_dirichlet_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain,
              dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);
            bct_for_bilinear_form_V2.partition(
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
              support_points_for_neumann_space_on_dirichlet_domain,
              support_points_for_neumann_space_on_neumann_domain,
              dof_average_cell_size_for_neumann_space_on_dirichlet_domain,
              dof_average_cell_size_for_neumann_space_on_neumann_domain);
            bct_for_bilinear_form_D2.partition(
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
              support_points_for_dirichlet_space_on_neumann_domain,
              support_points_for_dirichlet_space_on_dirichlet_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain,
              dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);
            bct_for_bilinear_form_K_prime2.partition(
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
              support_points_for_dirichlet_space_on_neumann_domain,
              support_points_for_neumann_space_on_neumann_domain,
              dof_average_cell_size_for_dirichlet_space_on_neumann_domain,
              dof_average_cell_size_for_neumann_space_on_neumann_domain);

            timer.stop();
            print_wall_time(deallog, timer, "build block cluster trees");

            /**
             * Initialize \hmatrices.
             */
            timer.start();

            V1_hmat =
              HMatrixSymm<spacedim>(bct_for_bilinear_form_V1, max_hmat_rank);

            std::cout << "=== Leaf set information of V1_hmat ===" << std::endl;
            V1_hmat.print_leaf_set_info(std::cout);

            K1_hmat =
              HMatrix<spacedim>(bct_for_bilinear_form_K1,
                                max_hmat_rank,
                                HMatrixSupport::Property::general,
                                HMatrixSupport::BlockType::diagonal_block);

            std::cout << "=== Leaf set information of K1_hmat ===" << std::endl;
            K1_hmat.print_leaf_set_info(std::cout);

            D1_hmat =
              HMatrixSymm<spacedim>(bct_for_bilinear_form_D1, max_hmat_rank);

            std::cout << "=== Leaf set information of D1_hmat ===" << std::endl;
            D1_hmat.print_leaf_set_info(std::cout);

            K2_hmat_with_mass_matrix =
              HMatrix<spacedim>(bct_for_bilinear_form_K2,
                                max_hmat_rank,
                                HMatrixSupport::Property::general,
                                HMatrixSupport::BlockType::diagonal_block);

            std::cout
              << "=== Leaf set information of K2_hmat_with_mass_matrix ==="
              << std::endl;
            K2_hmat_with_mass_matrix.print_leaf_set_info(std::cout);

            V2_hmat =
              HMatrix<spacedim>(bct_for_bilinear_form_V2,
                                max_hmat_rank,
                                HMatrixSupport::Property::general,
                                HMatrixSupport::BlockType::diagonal_block);

            std::cout << "=== Leaf set information of V2_hmat ===" << std::endl;
            V2_hmat.print_leaf_set_info(std::cout);

            D2_hmat =
              HMatrix<spacedim>(bct_for_bilinear_form_D2,
                                max_hmat_rank,
                                HMatrixSupport::Property::general,
                                HMatrixSupport::BlockType::diagonal_block);

            std::cout << "=== Leaf set information of D2_hmat ===" << std::endl;
            D2_hmat.print_leaf_set_info(std::cout);

            K_prime2_hmat_with_mass_matrix =
              HMatrix<spacedim>(bct_for_bilinear_form_K_prime2,
                                max_hmat_rank,
                                HMatrixSupport::Property::general,
                                HMatrixSupport::BlockType::diagonal_block);

            std::cout
              << "=== Leaf set information of K_prime2_hmat_with_mass_matrix ==="
              << std::endl;
            K_prime2_hmat_with_mass_matrix.print_leaf_set_info(std::cout);

            timer.stop();
            print_wall_time(deallog, timer, "initialize H-matrices");

            /**
             * Interpolate the Dirichlet boundary data on the extended Dirichlet
             * domain and set those unselected DoFs to be zero.
             */
            timer.start();

            dirichlet_bc.reinit(dof_handler_for_dirichlet_space.n_dofs());
            VectorTools::interpolate(dof_handler_for_dirichlet_space,
                                     *dirichlet_bc_functor_ptr,
                                     dirichlet_bc);

            for (types::global_dof_index i = 0; i < dirichlet_bc.size(); i++)
              {
                if (!dof_selectors_for_dirichlet_space_on_dirichlet_domain[i])
                  {
                    dirichlet_bc[i] = 0.;
                  }
              }

            /**
             * Extract the Dirichlet boundary data on the selected DoFs.
             */
            dirichlet_bc_on_selected_dofs.reinit(
              n_dofs_for_dirichlet_space_on_dirichlet_domain);
            for (types::global_dof_index i = 0;
                 i < n_dofs_for_dirichlet_space_on_dirichlet_domain;
                 i++)
              {
                dirichlet_bc_on_selected_dofs(i) = dirichlet_bc(
                  local_to_full_dirichlet_dof_indices_on_dirichlet_domain[i]);
              }

            /**
             * Permute the Dirichlet boundary data by following the mapping
             * from internal to external DoF numbering.
             */
            dirichlet_bc_internal_dof_numbering.reinit(
              n_dofs_for_dirichlet_space_on_dirichlet_domain);
            permute_vector(
              dirichlet_bc_on_selected_dofs,
              *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
              dirichlet_bc_internal_dof_numbering);

            /**
             * Interpolate the Neumann boundary data.
             */
            neumann_bc.reinit(dof_handler_for_neumann_space.n_dofs());
            VectorTools::interpolate(dof_handler_for_neumann_space,
                                     *neumann_bc_functor_ptr,
                                     neumann_bc);

            /**
             * Extract the Neumann boundary data on the selected DoFs.
             */
            neumann_bc_on_selected_dofs.reinit(
              n_dofs_for_neumann_space_on_neumann_domain);
            for (types::global_dof_index i = 0;
                 i < n_dofs_for_neumann_space_on_neumann_domain;
                 i++)
              {
                neumann_bc_on_selected_dofs(i) = neumann_bc(
                  local_to_full_neumann_dof_indices_on_neumann_domain[i]);
              }

            /**
             * Permute the Neumann boundary data by following the mapping
             * from internal to external DoF numbering.
             */
            neumann_bc_internal_dof_numbering.reinit(
              n_dofs_for_neumann_space_on_neumann_domain);
            permute_vector(
              neumann_bc_on_selected_dofs,
              *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
              neumann_bc_internal_dof_numbering);

            timer.stop();
            print_wall_time(deallog, timer, "interpolate boundary condition");

            /**
             * Allocate memory for the right-hand-side vectors and solution
             * vectors.
             */
            system_rhs_on_dirichlet_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain);
            system_rhs_on_neumann_domain.reinit(
              n_dofs_for_dirichlet_space_on_neumann_domain);
            system_rhs_on_combined_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain +
              n_dofs_for_dirichlet_space_on_neumann_domain);

            // N.B. This is the solution vector on all DoFs in the associated
            // DoF handler.
            neumann_data.reinit(dof_handler_for_neumann_space.n_dofs());
            dirichlet_data.reinit(dof_handler_for_dirichlet_space.n_dofs());

            // N.B. This is the solution vector on selected DoFs in the
            // associated DoF handler in the external DoF numbering.
            neumann_data_on_dirichlet_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain);
            dirichlet_data_on_neumann_domain.reinit(
              n_dofs_for_dirichlet_space_on_neumann_domain);

            neumann_data_on_dirichlet_domain_internal_dof_numbering.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain);
            // N.B. This is the solution vector on the selected DoFs in the
            // associated DoF handler in the internal DoF numbering.
            dirichlet_data_on_neumann_domain_internal_dof_numbering.reinit(
              n_dofs_for_dirichlet_space_on_neumann_domain);
            solution_on_combined_domain_internal_dof_numbering.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain +
              n_dofs_for_dirichlet_space_on_neumann_domain);

            break;
          }
          default: {
            Assert(false, ExcInternalError());

            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assign_dirichlet_bc(Function<spacedim> &functor)
  {
    dirichlet_bc_functor_ptr = &functor;
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assign_neumann_bc(Function<spacedim> &functor)
  {
    neumann_bc_functor_ptr = &functor;
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assemble_full_matrix_system()
  {
    MultithreadInfo::set_thread_limit(thread_num);

    switch (problem_type)
      {
          case DirichletBCProblem: {
            /**
             * Assemble the FEM scaled mass matrix, which is stored into the
             * full matrix for \f$K_2\f$.
             *
             * \mynote{The polynomial order specified for the Gauss-Legendre
             * quadrature rule for FEM integration is accurate for the
             * integration of \f$2N-1\f$-th polynomial, where \f$N\f is the
             * number of quadrature points in 1D.}
             */
            std::cout << "=== Assemble scaled mass matrix ===" << std::endl;

            /**
             * For the interior Laplace problem, \f$\frac{1}{2}I\f$ is
             * assembled, while for the exterior Laplace problem,
             * \f$-\frac{1}{2}I\f$ is assembled. It is also assumed that the
             * potential reference \f$u_0\f$ is zero when \f$\abs{x} \rightarrow
             * \infty\f$.
             */
            if (is_interior_problem)
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_neumann_space,
                  dof_handler_for_dirichlet_space,
                  0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K2_matrix_with_mass_matrix);
              }
            else
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_neumann_space,
                  dof_handler_for_dirichlet_space,
                  -0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K2_matrix_with_mass_matrix);
              }

            /**
             * Assemble the DLP matrix, which is added with the previous
             * scaled FEM mass matrix.
             */
            std::cout << "=== Assemble DLP matrix ===" << std::endl;
            assemble_bem_full_matrix(
              double_layer_kernel,
              1.0,
              dof_handler_for_neumann_space,
              dof_handler_for_dirichlet_space,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K2_matrix_with_mass_matrix);

            /**
             * Assemble the SLP matrix.
             */
            std::cout << "=== Assemble SLP matrix ===" << std::endl;
            assemble_bem_full_matrix(
              single_layer_kernel,
              1.0,
              dof_handler_for_neumann_space,
              dof_handler_for_neumann_space,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              V1_matrix);

            /**
             * Calculate the RHS vector.
             */
            K2_matrix_with_mass_matrix.vmult(system_rhs_on_dirichlet_domain,
                                             dirichlet_bc);

            break;
          }
          case NeumannBCProblem: {
            std::cout << "=== Assemble scaled mass matrix ===" << std::endl;

            /**
             * For the interior Laplace problem, \f$\frac{1}{2}I\f$ is
             * assembled, while for the exterior Laplace problem,
             * \f$-\frac{1}{2}I\f$ is assembled.
             */
            if (is_interior_problem)
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_dirichlet_space,
                  dof_handler_for_neumann_space,
                  0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K_prime2_matrix_with_mass_matrix);
              }
            else
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_dirichlet_space,
                  dof_handler_for_neumann_space,
                  -0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K_prime2_matrix_with_mass_matrix);
              }

            /**
             * Assemble the ADLP matrix, which is added with the
             * previous
             * scaled FEM mass matrix.
             */
            std::cout << "=== Assemble ADLP matrix ===" << std::endl;
            assemble_bem_full_matrix(
              adjoint_double_layer_kernel,
              -1.0,
              dof_handler_for_dirichlet_space,
              dof_handler_for_neumann_space,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K_prime2_matrix_with_mass_matrix);

            /**
             * Assemble the matrix for the hyper singular operator, where the
             * regularization method is adopted.
             */
            std::cout << "=== Assemble D matrix ===" << std::endl;

            assemble_bem_full_matrix(
              hyper_singular_kernel,
              1.0,
              dof_handler_for_dirichlet_space,
              dof_handler_for_dirichlet_space,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              D1_matrix);

            /**
             * Calculate the RHS vector.
             */
            K_prime2_matrix_with_mass_matrix.vmult(system_rhs_on_neumann_domain,
                                                   neumann_bc);

            /**
             * Solve the natural density.
             */
            solve_natural_density();

            /**
             * Calculate the vector \f$a\f$ in \f$\alpha a a^T\f$, where \f$a\f$
             * is the multiplication of the mass matrix and the natural density.
             */
            assemble_fem_mass_matrix_vmult<dim,
                                           spacedim,
                                           double,
                                           Vector<double>>(
              dof_handler_for_dirichlet_space,
              dof_handler_for_neumann_space,
              natural_density,
              QGauss<2>(fe_order_for_dirichlet_space + 1),
              mass_vmult_weq);

            /**
             * Add the matrix \f$\alpha a a^T\f$ into \f$D\f$.
             */
            FullMatrix<double> aaT(D1_matrix.m(), D1_matrix.n());
            aaT.outer_product(mass_vmult_weq, mass_vmult_weq);
            D1_matrix.add(alpha_for_neumann, aaT);

            break;
          }
          case MixedBCProblem: {
            Assert(false, ExcNotImplemented());

            break;
          }
          default: {
            Assert(false, ExcInternalError());

            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assemble_hmatrix_system()
  {
    LogStream::Prefix prefix_string("assemble_hmatrix_system");
#if ENABLE_NVTX == 1
    HierBEM::CUDAWrappers::NVTXRange nvtx_range("assemble_hmatrix_system");
#endif

    Timer timer;

    MultithreadInfo::set_thread_limit(thread_num);

    /**
     * Define the @p ACAConfig object.
     */
    ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

    switch (problem_type)
      {
          case DirichletBCProblem: {
#if ENABLE_MATRIX_EXPORT == 1
            // Output stream for matrices and vectors.
            std::ofstream out_mat;
            // Output stream for block cluster trees.
            std::ofstream out_bct;
#endif

            if (is_interior_problem)
              {
#if ENABLE_NVTX == 1
                HierBEM::CUDAWrappers::NVTXRange nvtx_range(
                  "assemble sigma*I+K");
#endif

                std::cout << "=== Assemble sigma*I+K ===" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K2_hmat_with_mass_matrix,
                  aca_config,
                  double_layer_kernel,
                  1.0,
                  0.5,
                  dof_to_cell_topo_for_neumann_space,
                  dof_to_cell_topo_for_dirichlet_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_neumann_space,
                  dof_handler_for_dirichlet_space,
                  nullptr,
                  nullptr,
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  kx_mapping_for_dirichlet_domain,
                  ky_mapping_for_dirichlet_domain,
                  *kx_mapping_data_for_dirichlet_domain,
                  *ky_mapping_data_for_dirichlet_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble sigma*I+K");
              }
            else
              {
#if ENABLE_NVTX == 1
                HierBEM::CUDAWrappers::NVTXRange nvtx_range(
                  "assemble (sigma-1)*I+K");
#endif

                std::cout << "=== Assemble (sigma-1)*I+K ===" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K2_hmat_with_mass_matrix,
                  aca_config,
                  double_layer_kernel,
                  1.0,
                  -0.5,
                  dof_to_cell_topo_for_neumann_space,
                  dof_to_cell_topo_for_dirichlet_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_neumann_space,
                  dof_handler_for_dirichlet_space,
                  nullptr,
                  nullptr,
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  kx_mapping_for_dirichlet_domain,
                  ky_mapping_for_dirichlet_domain,
                  *kx_mapping_data_for_dirichlet_domain,
                  *ky_mapping_data_for_dirichlet_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble (sigma-1)*I+K");
              }

#if ENABLE_MATRIX_EXPORT == 1
            // Print the RHS matrix.
            out_mat.open("matrices.dat");

            K2_hmat_with_mass_matrix.print_as_formatted_full_matrix(
              out_mat, "K", 15, true, 25);

            out_bct.open("K_bct.dat");
            K2_hmat_with_mass_matrix.write_leaf_set_by_iteration(out_bct,
                                                                 1e-12);
            out_bct.close();
#endif

            std::cout << "=== Assemble the RHS vector ===" << std::endl;

            timer.start();
            K2_hmat_with_mass_matrix.vmult(system_rhs_on_dirichlet_domain,
                                           dirichlet_bc_internal_dof_numbering,
                                           HMatrixSupport::Property::general);

            timer.stop();
            print_wall_time(deallog, timer, "assemble RHS vector");

#if ENABLE_MATRIX_EXPORT == 1
            // Print the RHS vector.
            print_vector_to_mat(out_mat,
                                "system_rhs",
                                system_rhs_on_dirichlet_domain,
                                false,
                                15,
                                25);
#endif

            // Release the RHS matrix.
            std::cout << "=== Release the RHS matrix ===" << std::endl;

            K2_hmat_with_mass_matrix.release();

            {
#if ENABLE_NVTX == 1
              HierBEM::CUDAWrappers::NVTXRange nvtx_range("assemble V");
#endif

              std::cout << "=== Assemble V ===" << std::endl;

              timer.start();

              fill_hmatrix_with_aca_plus_smp(
                thread_num,
                V1_hmat,
                aca_config,
                single_layer_kernel,
                1.0,
                dof_to_cell_topo_for_neumann_space,
                dof_to_cell_topo_for_neumann_space,
                SauterQuadratureRule<dim>(5, 4, 4, 3),
                dof_handler_for_neumann_space,
                dof_handler_for_neumann_space,
                nullptr,
                nullptr,
                *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                kx_mapping_for_dirichlet_domain,
                ky_mapping_for_dirichlet_domain,
                *kx_mapping_data_for_dirichlet_domain,
                *ky_mapping_data_for_dirichlet_domain,
                map_from_surface_mesh_to_volume_mesh,
                map_from_surface_mesh_to_volume_mesh,
                HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  SameTriangulations,
                true);

              timer.stop();
              print_wall_time(deallog, timer, "assemble V");
            }

#if ENABLE_MATRIX_EXPORT == 1
            V1_hmat.print_as_formatted_full_matrix(out_mat, "V", 15, true, 25);

            out_bct.open("V_bct.dat");
            V1_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();

            out_mat.close();
#endif

            break;
          }
          case NeumannBCProblem: {
#if ENABLE_MATRIX_EXPORT == 1
            // Output stream for matrices and vectors.
            std::ofstream out_mat;
            // Output stream for block cluster trees.
            std::ofstream out_bct;
#endif

            if (is_interior_problem)
              {
                std::cout << "=== Assemble (1-sigma)*I-K' ===" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K_prime2_hmat_with_mass_matrix,
                  aca_config,
                  adjoint_double_layer_kernel,
                  -1.0,
                  0.5,
                  dof_to_cell_topo_for_dirichlet_space,
                  dof_to_cell_topo_for_neumann_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_dirichlet_space,
                  dof_handler_for_neumann_space,
                  nullptr,
                  nullptr,
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  kx_mapping_for_neumann_domain,
                  ky_mapping_for_neumann_domain,
                  *kx_mapping_data_for_neumann_domain,
                  *ky_mapping_data_for_neumann_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble (1-sigma)*I-K'");
              }
            else
              {
                std::cout << "=== Assemble -sigma*I-K' ===" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K_prime2_hmat_with_mass_matrix,
                  aca_config,
                  adjoint_double_layer_kernel,
                  -1.0,
                  -0.5,
                  dof_to_cell_topo_for_dirichlet_space,
                  dof_to_cell_topo_for_neumann_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_dirichlet_space,
                  dof_handler_for_neumann_space,
                  nullptr,
                  nullptr,
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  kx_mapping_for_neumann_domain,
                  ky_mapping_for_neumann_domain,
                  *kx_mapping_data_for_neumann_domain,
                  *ky_mapping_data_for_neumann_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble -sigma*I-K'");
              }

#if ENABLE_MATRIX_EXPORT == 1
            // Print the RHS matrix.
            out_mat.open("matrices.dat");

            K_prime2_hmat_with_mass_matrix.print_as_formatted_full_matrix(
              out_mat, "K_prime", 15, true, 25);

            out_bct.open("K_prime_bct.dat");
            K_prime2_hmat_with_mass_matrix.write_leaf_set_by_iteration(out_bct,
                                                                       1e-12);
            out_bct.close();
#endif

            /**
             * Calculate the RHS vector.
             */
            std::cout << "=== Assemble the RHS vector ===" << std::endl;

            timer.start();

            K_prime2_hmat_with_mass_matrix.vmult(
              system_rhs_on_neumann_domain,
              neumann_bc_internal_dof_numbering,
              HMatrixSupport::Property::general);

            timer.stop();
            print_wall_time(deallog, timer, "assemble RHS vector");

#if ENABLE_MATRIX_EXPORT == 1
            // Print the RHS vector.
            print_vector_to_mat(out_mat,
                                "system_rhs",
                                system_rhs_on_neumann_domain,
                                false,
                                15,
                                25);
#endif

            std::cout << "=== Release K' ===" << std::endl;
            K_prime2_hmat_with_mass_matrix.release();

            /**
             * Solve the natural density.
             */
            std::cout << "=== Solve the natural density weq ===" << std::endl;

            timer.start();

            solve_natural_density();

            timer.stop();
            print_wall_time(deallog, timer, "solve natural density weq");

            /**
             * Calculate the vector \f$a\f$ in \f$\alpha a a^T\f$, where \f$a\f$
             * is the multiplication of the mass matrix and the natural density.
             */
            std::cout << "=== Calculate the vector a=M*weq ===" << std::endl;

            timer.start();

            assemble_fem_mass_matrix_vmult<dim,
                                           spacedim,
                                           double,
                                           Vector<double>>(
              dof_handler_for_dirichlet_space,
              dof_handler_for_neumann_space,
              natural_density,
              QGauss<2>(fe_order_for_dirichlet_space + 1),
              mass_vmult_weq);

            timer.stop();
            print_wall_time(deallog, timer, "calculate a=M*weq");

            /**
             * Assemble the regularized bilinear form for the hyper-singular
             * operator along with the stabilization term.
             */
            std::cout << "=== Assemble D ===" << std::endl;

            timer.start();

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              D1_hmat,
              aca_config,
              hyper_singular_kernel,
              1.0,
              mass_vmult_weq,
              alpha_for_neumann,
              dof_to_cell_topo_for_dirichlet_space,
              dof_to_cell_topo_for_dirichlet_space,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_dirichlet_space,
              dof_handler_for_dirichlet_space,
              nullptr,
              nullptr,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            timer.stop();
            print_wall_time(deallog, timer, "assemble D");

#if ENABLE_MATRIX_EXPORT == 1
            D1_hmat.print_as_formatted_full_matrix(out_mat, "D", 15, true, 25);

            out_bct.open("D_bct.dat");
            D1_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();

            out_mat.close();
#endif

            break;
          }
          case MixedBCProblem: {
            /**
             * For the mixed boundary condition, we firstly assemble the right
             * hand side matrices and vectors. Then after releasing these
             * matrices for saving the memory, we continue to assemble the left
             * hand side matrices, i.e. stiff matrices.
             */
#if ENABLE_MATRIX_EXPORT == 1
            // Output stream for matrices and vectors.
            std::ofstream out_mat;
            // Output stream for block cluster trees.
            std::ofstream out_bct;
#endif

            if (is_interior_problem)
              {
                std::cout << "=== Assemble sigma*I+K ===" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K2_hmat_with_mass_matrix,
                  aca_config,
                  double_layer_kernel,
                  1.0,
                  0.5,
                  dof_to_cell_topo_for_neumann_space,
                  dof_to_cell_topo_for_dirichlet_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_neumann_space,
                  dof_handler_for_dirichlet_space,
                  &local_to_full_neumann_dof_indices_on_dirichlet_domain,
                  &local_to_full_dirichlet_dof_indices_on_dirichlet_domain,
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  kx_mapping_for_dirichlet_domain,
                  ky_mapping_for_dirichlet_domain,
                  *kx_mapping_data_for_dirichlet_domain,
                  *ky_mapping_data_for_dirichlet_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble sigma*I+K");

                std::cout << "=== Assemble (1-sigma)*I-K' ===" << std::endl;

                timer.start();

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K_prime2_hmat_with_mass_matrix,
                  aca_config,
                  adjoint_double_layer_kernel,
                  -1.0,
                  0.5,
                  dof_to_cell_topo_for_dirichlet_space,
                  dof_to_cell_topo_for_neumann_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_dirichlet_space,
                  dof_handler_for_neumann_space,
                  &local_to_full_dirichlet_dof_indices_on_neumann_domain,
                  &local_to_full_neumann_dof_indices_on_neumann_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  kx_mapping_for_neumann_domain,
                  ky_mapping_for_neumann_domain,
                  *kx_mapping_data_for_neumann_domain,
                  *ky_mapping_data_for_neumann_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble (1-sigma)*I-K'");
              }
            else
              {
                std::cout << "=== Assemble (sigma-1)*I+K ===" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K2_hmat_with_mass_matrix,
                  aca_config,
                  double_layer_kernel,
                  1.0,
                  -0.5,
                  dof_to_cell_topo_for_neumann_space,
                  dof_to_cell_topo_for_dirichlet_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_neumann_space,
                  dof_handler_for_dirichlet_space,
                  &local_to_full_neumann_dof_indices_on_dirichlet_domain,
                  &local_to_full_dirichlet_dof_indices_on_dirichlet_domain,
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  kx_mapping_for_dirichlet_domain,
                  ky_mapping_for_dirichlet_domain,
                  *kx_mapping_data_for_dirichlet_domain,
                  *ky_mapping_data_for_dirichlet_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble (sigma-1)*I+K");

                std::cout << "=== Assemble -sigma*I-K' ===" << std::endl;

                timer.start();

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K_prime2_hmat_with_mass_matrix,
                  aca_config,
                  adjoint_double_layer_kernel,
                  -1.0,
                  -0.5,
                  dof_to_cell_topo_for_dirichlet_space,
                  dof_to_cell_topo_for_neumann_space,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_dirichlet_space,
                  dof_handler_for_neumann_space,
                  &local_to_full_dirichlet_dof_indices_on_neumann_domain,
                  &local_to_full_neumann_dof_indices_on_neumann_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  kx_mapping_for_neumann_domain,
                  ky_mapping_for_neumann_domain,
                  *kx_mapping_data_for_neumann_domain,
                  *ky_mapping_data_for_neumann_domain,
                  map_from_surface_mesh_to_volume_mesh,
                  map_from_surface_mesh_to_volume_mesh,
                  HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);

                timer.stop();
                print_wall_time(deallog, timer, "assemble -sigma*I-K'");
              }

            std::cout << "=== Assemble -V2 ===" << std::endl;

            timer.start();

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              V2_hmat,
              aca_config,
              single_layer_kernel,
              -1.0,
              dof_to_cell_topo_for_neumann_space,
              dof_to_cell_topo_for_neumann_space,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_neumann_space,
              dof_handler_for_neumann_space,
              &local_to_full_neumann_dof_indices_on_dirichlet_domain,
              &local_to_full_neumann_dof_indices_on_neumann_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              false);

            timer.stop();
            print_wall_time(deallog, timer, "assemble -V2");

            std::cout << "=== Assemble -D2 ===" << std::endl;

            timer.start();

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              D2_hmat,
              aca_config,
              hyper_singular_kernel,
              -1.0,
              dof_to_cell_topo_for_dirichlet_space,
              dof_to_cell_topo_for_dirichlet_space,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_dirichlet_space,
              dof_handler_for_dirichlet_space,
              &local_to_full_dirichlet_dof_indices_on_neumann_domain,
              &local_to_full_dirichlet_dof_indices_on_dirichlet_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              false);

            timer.stop();
            print_wall_time(deallog, timer, "assemble -D2");

#if ENABLE_MATRIX_EXPORT == 1
            // Print RHS matrices.
            out_mat.open("matrices.dat");

            K2_hmat_with_mass_matrix.print_as_formatted_full_matrix(
              out_mat, "K2", 15, true, 25);
            K_prime2_hmat_with_mass_matrix.print_as_formatted_full_matrix(
              out_mat, "K_prime2", 15, true, 25);
            V2_hmat.print_as_formatted_full_matrix(out_mat, "V2", 15, true, 25);
            D2_hmat.print_as_formatted_full_matrix(out_mat, "D2", 15, true, 25);

            out_bct.open("K2_bct.dat");
            K2_hmat_with_mass_matrix.write_leaf_set_by_iteration(out_bct,
                                                                 1e-12);
            out_bct.close();

            out_bct.open("K_prime2_bct.dat");
            K_prime2_hmat_with_mass_matrix.write_leaf_set_by_iteration(out_bct,
                                                                       1e-12);
            out_bct.close();

            out_bct.open("V2_bct.dat");
            V2_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();

            out_bct.open("D2_bct.dat");
            D2_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();
#endif

            // Calculate the RHS vectors in the mixed boundary value problem.
            std::cout << "=== Assemble RHS vectors ===" << std::endl;

            timer.start();

            K2_hmat_with_mass_matrix.vmult(system_rhs_on_dirichlet_domain,
                                           dirichlet_bc_internal_dof_numbering,
                                           HMatrixSupport::Property::general);
            V2_hmat.vmult(system_rhs_on_dirichlet_domain,
                          neumann_bc_internal_dof_numbering,
                          HMatrixSupport::Property::general);

            timer.stop();
            print_wall_time(deallog,
                            timer,
                            "assemble RHS vector on Dirichlet domain");

            timer.start();

            D2_hmat.vmult(system_rhs_on_neumann_domain,
                          dirichlet_bc_internal_dof_numbering,
                          HMatrixSupport::Property::general);
            K_prime2_hmat_with_mass_matrix.vmult(
              system_rhs_on_neumann_domain,
              neumann_bc_internal_dof_numbering,
              HMatrixSupport::Property::general);

            timer.stop();
            print_wall_time(deallog,
                            timer,
                            "assemble RHS vector on Neumann domain");

            // Combine the two part of RHS vectors.
            copy_vector(system_rhs_on_combined_domain,
                        0,
                        system_rhs_on_dirichlet_domain,
                        0,
                        system_rhs_on_dirichlet_domain.size());
            copy_vector(system_rhs_on_combined_domain,
                        system_rhs_on_dirichlet_domain.size(),
                        system_rhs_on_neumann_domain,
                        0,
                        system_rhs_on_neumann_domain.size());

#if ENABLE_MATRIX_EXPORT == 1
            // Print RHS vectors.
            print_vector_to_mat(out_mat,
                                "system_rhs_on_combined_domain",
                                system_rhs_on_combined_domain,
                                false,
                                15,
                                25);

            print_vector_to_mat(out_mat,
                                "system_rhs_on_dirichlet_domain",
                                system_rhs_on_dirichlet_domain,
                                false,
                                15,
                                25);

            print_vector_to_mat(out_mat,
                                "system_rhs_on_neumann_domain",
                                system_rhs_on_neumann_domain,
                                false,
                                15,
                                25);
#endif

            // Release the RHS matrices.
            std::cout << "=== Release RHS matrices ===" << std::endl;

            K2_hmat_with_mass_matrix.release();
            K_prime2_hmat_with_mass_matrix.release();
            V2_hmat.release();
            D2_hmat.release();

            std::cout << "=== Assemble V1 ===" << std::endl;

            timer.start();

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              V1_hmat,
              aca_config,
              single_layer_kernel,
              1.0,
              dof_to_cell_topo_for_neumann_space,
              dof_to_cell_topo_for_neumann_space,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_neumann_space,
              dof_handler_for_neumann_space,
              &local_to_full_neumann_dof_indices_on_dirichlet_domain,
              &local_to_full_neumann_dof_indices_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            timer.stop();
            print_wall_time(deallog, timer, "assemble V1");

            std::cout << "=== Assemble -K1 ===" << std::endl;

            timer.start();

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              K1_hmat,
              aca_config,
              double_layer_kernel,
              -1.0,
              dof_to_cell_topo_for_neumann_space,
              dof_to_cell_topo_for_dirichlet_space,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_neumann_space,
              dof_handler_for_dirichlet_space,
              &local_to_full_neumann_dof_indices_on_dirichlet_domain,
              &local_to_full_dirichlet_dof_indices_on_neumann_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              false);

            timer.stop();
            print_wall_time(deallog, timer, "assemble -K1");

            std::cout << "=== Assemble D1 ===" << std::endl;

            timer.start();

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              D1_hmat,
              aca_config,
              hyper_singular_kernel,
              1.0,
              dof_to_cell_topo_for_dirichlet_space,
              dof_to_cell_topo_for_dirichlet_space,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_dirichlet_space,
              dof_handler_for_dirichlet_space,
              &local_to_full_dirichlet_dof_indices_on_neumann_domain,
              &local_to_full_dirichlet_dof_indices_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_surface_mesh_to_volume_mesh,
              map_from_surface_mesh_to_volume_mesh,
              HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            timer.stop();
            print_wall_time(deallog, timer, "assemble D1");

            // Assemble the block matrix.
            std::cout << "=== Assemble system block matrix ===" << std::endl;
            M_hmat =
              HBlockMatrixSkewSymm<spacedim>(&V1_hmat, &K1_hmat, &D1_hmat);

#if ENABLE_MATRIX_EXPORT == 1
            // Print LHS matrices.
            V1_hmat.print_as_formatted_full_matrix(out_mat, "V1", 15, true, 25);
            K1_hmat.print_as_formatted_full_matrix(out_mat, "K1", 15, true, 25);
            D1_hmat.print_as_formatted_full_matrix(out_mat, "D1", 15, true, 25);

            out_bct.open("V1_bct.dat");
            V1_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();

            out_bct.open("K1_bct.dat");
            K1_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();

            out_bct.open("D1_bct.dat");
            D1_hmat.write_leaf_set_by_iteration(out_bct, 1e-12);
            out_bct.close();

            out_mat.close();
#endif

            break;
          }
          default: {
            Assert(false, ExcInternalError());

            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assemble_hmatrix_preconditioner()
  {
    LogStream::Prefix prefix_string("assemble_hmatrix_preconditioner");

    Timer timer;

    MultithreadInfo::set_thread_limit(thread_num);

    switch (problem_type)
      {
          case DirichletBCProblem: {
            std::cout << "=== Assemble preconditioner for V ===" << std::endl;

            // Directly make a copy of the existing \hmat and then truncate
            // its rank.
            V1_hmat_preconditioner = V1_hmat;

            V1_hmat_preconditioner.truncate_to_rank_preserve_positive_definite(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "truncate V");

            /**
             * Perform Cholesky factorisation of the preconditioner.
             */
            std::cout
              << "=== Cholesky factorization of the preconditioner for V ==="
              << std::endl;

            timer.start();

            V1_hmat_preconditioner.compute_cholesky_factorization(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "Cholesky factorization of V");

            break;
          }
          case NeumannBCProblem: {
            std::cout << "=== Assemble preconditioner for D ===" << std::endl;

            // Directly make a copy of the existing \hmat and then truncate
            // its rank.
            D1_hmat_preconditioner = D1_hmat;
            D1_hmat_preconditioner.truncate_to_rank_preserve_positive_definite(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "truncate D");

            /**
             * Perform Cholesky factorization of the preconditioner.
             */
            std::cout << "=== Cholesky factorization of D ===" << std::endl;

            timer.start();

            D1_hmat_preconditioner.compute_cholesky_factorization(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "Cholesky factorization of D");

            break;
          }
          case MixedBCProblem: {
            // Assemble preconditioners for the mixed boundary value problem.
            std::cout
              << "=== Assemble preconditioner for the system block matrix ==="
              << std::endl;

            // Directly make copies of existing \hmatrices and then truncate
            // their ranks.
            M11_in_preconditioner = V1_hmat;
            M12_in_preconditioner = K1_hmat;
            M22_in_preconditioner = D1_hmat;

            M11_in_preconditioner.truncate_to_rank_preserve_positive_definite(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "truncate M11(==V1)");

            timer.start();

            M12_in_preconditioner.truncate_to_rank(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "truncate M12(==K1)");

            timer.start();

            M22_in_preconditioner.truncate_to_rank_preserve_positive_definite(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "truncate M22(==D1)");

            M_hmat_preconditioner =
              HBlockMatrixSkewSymmPreconditioner<spacedim>(
                &M11_in_preconditioner,
                &M12_in_preconditioner,
                &M22_in_preconditioner);

            // Perform \hmat-LU factorization to the block preconditioner.
            std::cout << "=== LU factorization of system block matrix ==="
                      << std::endl;

            timer.start();

            M_hmat_preconditioner.compute_lu_factorization(
              max_hmat_rank_for_preconditioner);

            timer.stop();
            print_wall_time(deallog, timer, "LU factorization of M");

            break;
          }
          default: {
            Assert(false, ExcInternalError());
            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::solve()
  {
#if ENABLE_NVTX == 1
    HierBEM::CUDAWrappers::NVTXRange nvtx_range("solve");
#endif

    std::cout << "=== Solve problem ===" << std::endl;

    if (!is_use_hmat)
      {
        switch (problem_type)
          {
              case DirichletBCProblem: {
                SolverControl solver_control(1000, 1e-12, true, true);
                SolverCG<>    solver(solver_control);

                solver.solve(V1_matrix,
                             neumann_data,
                             system_rhs_on_dirichlet_domain,
                             PreconditionIdentity());

                break;
              }
              case NeumannBCProblem: {
                SolverControl solver_control(1000, 1e-12, true, true);
                SolverCG<>    solver(solver_control);

                solver.solve(D1_matrix,
                             dirichlet_data,
                             system_rhs_on_neumann_domain,
                             PreconditionIdentity());

                break;
              }
              case MixedBCProblem: {
                Assert(false, ExcNotImplemented());

                break;
              }
              default: {
                Assert(false, ExcInternalError());
              }
          }
      }
    else
      {
        switch (problem_type)
          {
              case DirichletBCProblem: {
                SolverControl solver_control(1000, 1e-12, true, true);
                SolverCG<Vector<double>> solver(solver_control);

                solver.solve(
                  V1_hmat,
                  neumann_data_on_dirichlet_domain_internal_dof_numbering,
                  system_rhs_on_dirichlet_domain,
                  V1_hmat_preconditioner);

                /**
                 * Permute the solution vector by following the mapping
                 * from external to internal DoF numbering.
                 */
                permute_vector(
                  neumann_data_on_dirichlet_domain_internal_dof_numbering,
                  *dof_e2i_numbering_for_neumann_space_on_dirichlet_domain,
                  neumann_data);

                break;
              }
              case NeumannBCProblem: {
                SolverControl solver_control(1000, 1e-12, true, true);
                SolverCG<Vector<double>> solver(solver_control);

                solver.solve(
                  D1_hmat,
                  dirichlet_data_on_neumann_domain_internal_dof_numbering,
                  system_rhs_on_neumann_domain,
                  D1_hmat_preconditioner);

                /**
                 * Permute the solution vector by following the mapping
                 * from external to internal DoF numbering.
                 */
                permute_vector(
                  dirichlet_data_on_neumann_domain_internal_dof_numbering,
                  *dof_e2i_numbering_for_dirichlet_space_on_neumann_domain,
                  dirichlet_data);

                break;
              }
              case MixedBCProblem: {
                SolverControl solver_control(1000, 1e-12, true, true);
                SolverBicgstab<Vector<double>> solver(solver_control);

                solver.solve(M_hmat,
                             solution_on_combined_domain_internal_dof_numbering,
                             system_rhs_on_combined_domain,
                             M_hmat_preconditioner);

                // Split the solution vector
                copy_vector(
                  neumann_data_on_dirichlet_domain_internal_dof_numbering,
                  0,
                  solution_on_combined_domain_internal_dof_numbering,
                  0,
                  neumann_data_on_dirichlet_domain_internal_dof_numbering
                    .size());
                copy_vector(
                  dirichlet_data_on_neumann_domain_internal_dof_numbering,
                  0,
                  solution_on_combined_domain_internal_dof_numbering,
                  neumann_data_on_dirichlet_domain_internal_dof_numbering
                    .size(),
                  dirichlet_data_on_neumann_domain_internal_dof_numbering
                    .size());

                // Permute the solution vector by following the mapping from
                // external to internal DoF numbering.
                permute_vector(
                  neumann_data_on_dirichlet_domain_internal_dof_numbering,
                  *dof_e2i_numbering_for_neumann_space_on_dirichlet_domain,
                  neumann_data_on_dirichlet_domain);
                permute_vector(
                  dirichlet_data_on_neumann_domain_internal_dof_numbering,
                  *dof_e2i_numbering_for_dirichlet_space_on_neumann_domain,
                  dirichlet_data_on_neumann_domain);

                // Combine the solution vector and the boundary condition vector
                // to form a complete Cauchy data.
                DoFToolsExt::extend_selected_dof_values_to_full_dofs(
                  dirichlet_data,
                  dirichlet_data_on_neumann_domain,
                  local_to_full_dirichlet_dof_indices_on_neumann_domain);
                DoFToolsExt::extend_selected_dof_values_to_full_dofs(
                  dirichlet_data,
                  dirichlet_bc_on_selected_dofs,
                  local_to_full_dirichlet_dof_indices_on_dirichlet_domain);

                DoFToolsExt::extend_selected_dof_values_to_full_dofs(
                  neumann_data,
                  neumann_data_on_dirichlet_domain,
                  local_to_full_neumann_dof_indices_on_dirichlet_domain);
                DoFToolsExt::extend_selected_dof_values_to_full_dofs(
                  neumann_data,
                  neumann_bc_on_selected_dofs,
                  local_to_full_neumann_dof_indices_on_neumann_domain);

                break;
              }
              default: {
                Assert(false, ExcInternalError());
              }
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::output_results() const
  {
    std::cout << "=== Output results ===" << std::endl;

    std::ofstream          vtk_output;
    DataOut<dim, spacedim> data_out;

    switch (problem_type)
      {
          case DirichletBCProblem: {
            vtk_output.open("solution_for_dirichlet_bc.vtk",
                            std::ofstream::out);

            data_out.add_data_vector(dof_handler_for_neumann_space,
                                     neumann_data,
                                     "neumann_data");
            data_out.add_data_vector(dof_handler_for_dirichlet_space,
                                     dirichlet_bc,
                                     "dirichlet_data");

            data_out.build_patches();
            data_out.write_vtk(vtk_output);

            print_vector_to_mat(std::cout, "solution", neumann_data, false);

            break;
          }
          case NeumannBCProblem: {
            vtk_output.open("solution_for_neumann_bc.vtk", std::ofstream::out);

            data_out.add_data_vector(dof_handler_for_dirichlet_space,
                                     dirichlet_data,
                                     "dirichlet_data");
            data_out.add_data_vector(dof_handler_for_neumann_space,
                                     neumann_bc,
                                     "neumann_data");

            data_out.build_patches();
            data_out.write_vtk(vtk_output);

            print_vector_to_mat(std::cout, "solution", dirichlet_data, false);

            break;
          }
          case MixedBCProblem: {
            vtk_output.open("solution_for_mixed_bc.vtk", std::ofstream::out);

            data_out.add_data_vector(dof_handler_for_neumann_space,
                                     neumann_data,
                                     "neumann_data");
            data_out.add_data_vector(dof_handler_for_dirichlet_space,
                                     dirichlet_data,
                                     "dirichlet_data");
            data_out.build_patches();
            data_out.write_vtk(vtk_output);

            print_vector_to_mat(
              std::cout,
              "solution_on_combined_domain_internal_dof_numbering",
              solution_on_combined_domain_internal_dof_numbering,
              false,
              15,
              25);

            print_vector_to_mat(std::cout,
                                "solution_on_dirichlet_domain",
                                neumann_data,
                                false,
                                15,
                                25);

            print_vector_to_mat(std::cout,
                                "solution_on_neumann_domain",
                                dirichlet_data,
                                false,
                                15,
                                25);

            break;
          }
          default: {
            Assert(false, ExcInternalError());
          }
      }

    vtk_output.close();
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::output_potential_at_target_points() const
  {
    /**
     * Create a plane of regular grid for potential evaluation.
     */
    std::vector<Point<spacedim, double>> potential_grid_points;

    // Number of cells in the X direction.
    unsigned int nx = 100;
    // Number of cells in the Y direction.
    unsigned int ny = 100;
    // Plane dimension in the X direction.
    double grid_plane_xdim = 4.0;
    // Plane dimension in the Y direction.
    double grid_plane_ydim = 4.0;
    // Grid point spacing in the X direction.
    double dx = grid_plane_xdim / nx;
    // Grid point spacing in the Y direction.
    double dy = grid_plane_ydim / ny;
    // Shift in the X direction.
    double x_shift = -grid_plane_xdim / 2;
    // Shift in the Y direction.
    double y_shift = -grid_plane_ydim / 2;
    // Plane Z height
    double z = 4.0;

    /**
     * For visualization in GNU Octave using @p surf, here we generate
     * the list of grid points with their X coordinate components
     * running the fastest.
     */
    for (unsigned int j = 0; j <= ny; j++)
      {
        for (unsigned int i = 0; i <= nx; i++)
          {
            potential_grid_points.push_back(
              Point<spacedim, double>(i * dx + x_shift, j * dy + y_shift, z));
          }
      }

    /**
     * Vector storing the list of evaluated potential values.
     */
    Vector<double> potential_values(potential_grid_points.size());

    switch (problem_type)
      {
          case DirichletBCProblem: {
            if (is_interior_problem)
              {
                /**
                 * Evaluate the double layer potential, which is the negated
                 * double layer potential integral operator applied to the
                 * Dirichlet data. \f[
                 * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
                 * int} u(y) \intd s_y
                 * \f]
                 */
                std::cout << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(double_layer_kernel,
                                             -1.0,
                                             dof_handler_for_dirichlet_space,
                                             dirichlet_bc,
                                             false,
                                             potential_grid_points,
                                             potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cout << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(single_layer_kernel,
                                             1.0,
                                             dof_handler_for_neumann_space,
                                             neumann_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);
              }
            else
              {
                /**
                 * Evaluate the double layer potential, which is the negated
                 * double layer potential integral operator applied to the
                 * Dirichlet data. \f[
                 * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
                 * int} u(y) \intd s_y
                 * \f]
                 */
                std::cout << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(double_layer_kernel,
                                             1.0,
                                             dof_handler_for_dirichlet_space,
                                             dirichlet_bc,
                                             false,
                                             potential_grid_points,
                                             potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cout << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(single_layer_kernel,
                                             -1.0,
                                             dof_handler_for_neumann_space,
                                             neumann_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);
              }

            break;
          }
          case NeumannBCProblem: {
            if (is_interior_problem)
              {
                /**
                 * Evaluate the double layer potential, which is the negated
                 * double layer potential integral operator applied to the
                 * Dirichlet data. \f[
                 * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
                 * int} u(y) \intd s_y
                 * \f]
                 */
                std::cout << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(double_layer_kernel,
                                             -1.0,
                                             dof_handler_for_dirichlet_space,
                                             dirichlet_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cout << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(single_layer_kernel,
                                             1.0,
                                             dof_handler_for_neumann_space,
                                             neumann_bc,
                                             false,
                                             potential_grid_points,
                                             potential_values);
              }
            else
              {
                /**
                 * Evaluate the double layer potential, which is the negated
                 * double layer potential integral operator applied to the
                 * Dirichlet data. \f[
                 * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
                 * int} u(y) \intd s_y
                 * \f]
                 */
                std::cout << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(double_layer_kernel,
                                             1.0,
                                             dof_handler_for_dirichlet_space,
                                             dirichlet_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cout << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(single_layer_kernel,
                                             -1.0,
                                             dof_handler_for_neumann_space,
                                             neumann_bc,
                                             false,
                                             potential_grid_points,
                                             potential_values);
              }

            break;
          }
          case MixedBCProblem: {
            if (is_interior_problem)
              {
                /**
                 * Evaluate the double layer potential, which is the negated
                 * double layer potential integral operator applied to the
                 * Dirichlet data. \f[
                 * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
                 * int} u(y) \intd s_y
                 * \f]
                 */
                std::cout << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(double_layer_kernel,
                                             -1.0,
                                             dof_handler_for_dirichlet_space,
                                             dirichlet_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cout << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(single_layer_kernel,
                                             1.0,
                                             dof_handler_for_neumann_space,
                                             neumann_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);
              }
            else
              {
                /**
                 * Evaluate the double layer potential, which is the negated
                 * double layer potential integral operator applied to the
                 * Dirichlet data. \f[
                 * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
                 * int} u(y) \intd s_y
                 * \f]
                 */
                std::cout << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(double_layer_kernel,
                                             1.0,
                                             dof_handler_for_dirichlet_space,
                                             dirichlet_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cout << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(single_layer_kernel,
                                             -1.0,
                                             dof_handler_for_neumann_space,
                                             neumann_data,
                                             false,
                                             potential_grid_points,
                                             potential_values);
              }

            break;
          }
          default: {
            Assert(false, ExcInternalError());
            break;
          }
      }

    print_vector_to_mat(std::cout, "potential_values", potential_values, false);
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::run()
  {
    LogStream::Prefix prefix_string("run");
#if ENABLE_NVTX == 1
    HierBEM::CUDAWrappers::NVTXRange nvtx_range("run");
#endif

    Timer timer;
    setup_system();
    timer.stop();
    print_wall_time(deallog, timer, "setup system");

    if (!is_use_hmat)
      {
        timer.start();
        assemble_full_matrix_system();
        timer.stop();
        print_wall_time(deallog, timer, "assemble full matrix system");
      }
    else
      {
        timer.start();
        assemble_hmatrix_system();
        timer.stop();
        print_wall_time(deallog, timer, "assemble H-matrix system");

        timer.start();
        assemble_hmatrix_preconditioner();
        timer.stop();
        print_wall_time(deallog, timer, "assemble H-matrix preconditioner");
      }

    timer.start();
    solve();
    timer.stop();
    print_wall_time(deallog, timer, "solve equation");

    timer.start();
    output_results();
    timer.stop();
    print_wall_time(deallog, timer, "output results");
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::verify_neumann_solution_in_space()
  {
    /**
     * Calculate the inner product of the solution and the natural density.
     * Because the numerical solution is sought in the Sobolev space
     * \f$H^{-1/2}_*(\Gamma)\f$, it is orthogonal to the natural density
     * \f$w_{\rm eq}\f$. The analytical solution does not obey this constraint.
     */
    Vector<double> v(dof_handler_for_dirichlet_space.n_dofs());
    assemble_fem_mass_matrix_vmult<dim, spacedim, double, Vector<double>>(
      dof_handler_for_dirichlet_space,
      dof_handler_for_neumann_space,
      natural_density,
      QGauss<2>(fe_order_for_dirichlet_space + 1),
      v);
    std::cout << "Analytical solution <gamma_0 u, weq>="
              << analytical_solution_on_neumann_domain * v << "\n";
    std::cout << "Numerical solution <gamma_0 u, weq>=" << dirichlet_data * v
              << std::endl;
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::initialize_mapping_data()
  {
    /**
     * Initialize the @p InternalData objects in the mapping objects for the
     * Dirichlet domain.
     */
    if (problem_type == DirichletBCProblem || problem_type == MixedBCProblem)
      {
        /**
         * Create the internal data object in the parent @p Mapping object.
         *
         * N.B. A dummy quadrature object is passed to the @p get_data function. The
         * @p UpdateFlags is set to @p update_default (it means no update),
         * which at the moment disables any memory allocation, because this
         * operation will be manually taken care of later on.
         */
        std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
          kx_mapping_database_for_dirichlet_domain =
            kx_mapping_for_dirichlet_domain.get_data(update_default,
                                                     QGauss<dim>(1));

        /**
         * Downcast the smart pointer of @p Mapping<dim, spacedim>::InternalDataBase to
         * @p MappingQGeneric<dim,spacedim>::InternalData by first unwrapping
         * the original smart pointer via @p static_cast then wrapping it again.
         */
        kx_mapping_data_for_dirichlet_domain = std::unique_ptr<
          typename MappingQGeneric<dim, spacedim>::InternalData>(
          static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
            kx_mapping_database_for_dirichlet_domain.release()));

        std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
          ky_mapping_database_for_dirichlet_domain =
            ky_mapping_for_dirichlet_domain.get_data(update_default,
                                                     QGauss<dim>(1));

        ky_mapping_data_for_dirichlet_domain = std::unique_ptr<
          typename MappingQGeneric<dim, spacedim>::InternalData>(
          static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
            ky_mapping_database_for_dirichlet_domain.release()));
      }

    /**
     * Initialize the @p InternalData objects in the mapping objects for the
     * Neumann domain.
     */
    if (problem_type == NeumannBCProblem || problem_type == MixedBCProblem)
      {
        std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
          kx_mapping_database_for_neumann_domain =
            kx_mapping_for_neumann_domain.get_data(update_default,
                                                   QGauss<dim>(1));

        kx_mapping_data_for_neumann_domain = std::unique_ptr<
          typename MappingQGeneric<dim, spacedim>::InternalData>(
          static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
            kx_mapping_database_for_neumann_domain.release()));

        std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
          ky_mapping_database_for_neumann_domain =
            ky_mapping_for_neumann_domain.get_data(update_default,
                                                   QGauss<dim>(1));

        ky_mapping_data_for_neumann_domain = std::unique_ptr<
          typename MappingQGeneric<dim, spacedim>::InternalData>(
          static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
            ky_mapping_database_for_neumann_domain.release()));
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::solve_natural_density()
  {
    Assert(problem_type == ProblemType::NeumannBCProblem, ExcInternalError());

    if (!is_use_hmat)
      {
        /**
         * Assemble the SLP matrix which is used for solving the natural
         * density \f$w_{\rm eq}\f$.
         */
        std::cout << "=== Assemble V for solving the natural density === "
                  << std::endl;

        assemble_bem_full_matrix(
          single_layer_kernel,
          1.0,
          dof_handler_for_neumann_space,
          dof_handler_for_neumann_space,
          kx_mapping_for_neumann_domain,
          ky_mapping_for_neumann_domain,
          *kx_mapping_data_for_neumann_domain,
          *ky_mapping_data_for_neumann_domain,
          map_from_surface_mesh_to_volume_mesh,
          map_from_surface_mesh_to_volume_mesh,
          HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
            SameTriangulations,
          SauterQuadratureRule<dim>(5, 4, 4, 3),
          V1_matrix);
      }
    else
      {
        /**
         * Define the @p ACAConfig object.
         */
        ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

        std::cout << "=== Assemble V for solving the natural density ==="
                  << std::endl;

        fill_hmatrix_with_aca_plus_smp(
          thread_num,
          V1_hmat,
          aca_config,
          single_layer_kernel,
          1.0,
          dof_to_cell_topo_for_neumann_space,
          dof_to_cell_topo_for_neumann_space,
          SauterQuadratureRule<dim>(5, 4, 4, 3),
          dof_handler_for_neumann_space,
          dof_handler_for_neumann_space,
          nullptr,
          nullptr,
          *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
          *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
          kx_mapping_for_neumann_domain,
          ky_mapping_for_neumann_domain,
          *kx_mapping_data_for_neumann_domain,
          *ky_mapping_data_for_neumann_domain,
          map_from_surface_mesh_to_volume_mesh,
          map_from_surface_mesh_to_volume_mesh,
          HierBEM::BEMTools::DetectCellNeighboringTypeMethod::
            SameTriangulations,
          true);
      }

    /**
     * Assemble the RHS vector for solving the natural density \f$w_{\rm
     * eq}\f$.
     */
    std::cout << "=== Assemble the RHS vector for natural density ==="
              << std::endl;

    assemble_rhs_linear_form_vector(1.0,
                                    dof_handler_for_neumann_space,
                                    QGauss<2>(fe_for_neumann_space.degree + 1),
                                    system_rhs_for_natural_density);

    /**
     * Solve the natural density \f$w_{\rm eq}\f$.
     */
    SolverControl            solver_control(1000, 1e-6, true, true);
    SolverCG<Vector<double>> solver(solver_control);

    if (!is_use_hmat)
      {
        solver.solve(V1_matrix,
                     natural_density,
                     system_rhs_for_natural_density,
                     PreconditionIdentity());
      }
    else
      {
        std::cout << "=== Assemble preconditioner for the V matrix ==="
                  << std::endl;

        V1_hmat_preconditioner = V1_hmat;
        V1_hmat_preconditioner.truncate_to_rank_preserve_positive_definite(
          max_hmat_rank_for_preconditioner);

        /**
         * Perform Cholesky factorisation of the preconditioner.
         */
        std::cout << "=== Cholesky factorization of V ===" << std::endl;
        V1_hmat_preconditioner.compute_cholesky_factorization(
          max_hmat_rank_for_preconditioner);

        solver.solve(V1_hmat,
                     natural_density,
                     system_rhs_for_natural_density,
                     V1_hmat_preconditioner);

        std::cout << "=== Release V and its preconditioner ===" << std::endl;
        V1_hmat.release();
        V1_hmat_preconditioner.release();
      }

    /**
     * Calculate the stabilization factor \f$\alpha\f$, which is the inner
     * product of \f$w_{\rm eq}\f$ and the RHS vector \f$\langle 1, \psi_i
     * \rangle\f$.
     */
    alpha_for_neumann =
      1.0 / 4.0 / (natural_density * system_rhs_for_natural_density);
    std::cout << "Neumann stabilization factor: " << alpha_for_neumann
              << std::endl;

    std::cout << "=== Release the RHS vector for solving natural density ==="
              << std::endl;
    system_rhs_for_natural_density.reinit(0);
  }
} // namespace HierBEM

#endif /* INCLUDE_LAPLACE_BEM_H_ */
