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
#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "aca_plus.h"
#include "bem_general.h"
#include "bem_tools.h"
#include "bem_values.h"
#include "block_cluster_tree.h"
#include "cluster_tree.h"
#include "hmatrix.h"
#include "hmatrix_symm.h"
#include "hmatrix_symm_preconditioner.h"
#include "laplace_kernels.h"
#include "mapping_q_generic_ext.h"
#include "quadrature.templates.h"

namespace IdeoBEM
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
      const Quadrature<dim1> &           quad_rule,
      MatrixType &                       target_full_matrix);

    template <int dim1,
              int spacedim1,
              typename RangeNumberType,
              typename MatrixType>
    friend void
    assemble_bem_full_matrix(
      const KernelFunction<spacedim1, RangeNumberType> &kernel,
      const DoFHandler<dim1, spacedim1> &  dof_handler_for_test_space,
      const DoFHandler<dim1, spacedim1> &  dof_handler_for_trial_space,
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
      const SauterQuadratureRule<dim1> &    sauter_quad_rule,
      MatrixType &                          target_full_matrix);

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
     * TODO Initialize boundary indices which are used to specify Dirichlet
     * domain and Neumann domain.
     */
    void
    initialize_boundary_ids();

    /**
     * Extract the boundary mesh from the volume mesh for BEM.
     */
    void
    extract_boundary_mesh();

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
    output_results();

    void
    output_potential_at_target_points();

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

    /**
     * Triangulation for the Dirichlet domain.
     */
    Triangulation<dim, spacedim> triangulation_for_dirichlet_domain;

    /**
     * Triangulation for the Neumann domain.
     */
    Triangulation<dim, spacedim> triangulation_for_neumann_domain;

    /**
     * A set of boundary indices for the Dirichlet domain.
     */
    std::set<types::boundary_id> boundary_ids_for_dirichlet_domain;

    /**
     * A set of boundary indices for the Neumann domain.
     */
    std::set<types::boundary_id> boundary_ids_for_neumann_domain;

    /**
     * Map from cell iterators in the surface mesh for the Dirichlet domain to
     * the face iterators in the original volume mesh.
     */
    std::map<typename Triangulation<dim, spacedim>::cell_iterator,
             typename Triangulation<dim + 1, spacedim>::face_iterator>
      map_from_dirichlet_boundary_mesh_to_volume_mesh;

    /**
     * Map from cell iterators in the surface mesh for the Neumann domain to
     * the face iterators in the original volume mesh.
     */
    std::map<typename Triangulation<dim, spacedim>::cell_iterator,
             typename Triangulation<dim + 1, spacedim>::face_iterator>
      map_from_neumann_boundary_mesh_to_volume_mesh;

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
    DoFHandler<dim, spacedim>
                              dof_handler_for_dirichlet_space_on_dirichlet_domain;
    DoFHandler<dim, spacedim> dof_handler_for_dirichlet_space_on_neumann_domain;
    DoFHandler<dim, spacedim> dof_handler_for_neumann_space_on_dirichlet_domain;
    DoFHandler<dim, spacedim> dof_handler_for_neumann_space_on_neumann_domain;

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

    /**
     * DoF-to-cell topologies for various DoF handlers, which are used for
     * matrix assembly on a pair of DoFs.
     */
    std::vector<std::vector<unsigned int>>
      dof_to_cell_topo_for_dirichlet_space_on_dirichlet_domain;
    std::vector<std::vector<unsigned int>>
      dof_to_cell_topo_for_dirichlet_space_on_neumann_domain;
    std::vector<std::vector<unsigned int>>
      dof_to_cell_topo_for_neumann_space_on_dirichlet_domain;
    std::vector<std::vector<unsigned int>>
      dof_to_cell_topo_for_neumann_space_on_neumann_domain;

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
    LaplaceKernel::SingleLayerKernel<3> single_layer_kernel;
    /**
     * Kernel function for the double layer potential.
     */
    LaplaceKernel::DoubleLayerKernel<3> double_layer_kernel;
    /**
     * Kernel function for the adjoint double layer potential.
     */
    LaplaceKernel::AdjointDoubleLayerKernel<3> adjoint_double_layer_kernel;
    /**
     * Kernel function for the hyper-singular potential.
     */
    LaplaceKernel::HyperSingularKernelRegular<3> hyper_singular_kernel;

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
    HMatrixSymm<spacedim> V1_hmat;
    HMatrix<spacedim>     K1_hmat;
    HMatrix<spacedim>     K_prime1_hmat;
    HMatrixSymm<spacedim> D1_hmat;
    HMatrix<spacedim>     V2_hmat;
    HMatrix<spacedim>     K2_hmat_with_mass_matrix;
    HMatrix<spacedim>     K_prime2_hmat_with_mass_matrix;
    HMatrix<spacedim>     D2_hmat;

    /**
     * Preconditioners
     */
    HMatrixSymmPreconditioner<spacedim> V1_hmat_preconditioner;
    HMatrixSymmPreconditioner<spacedim> D1_hmat_preconditioner;

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
     * Neumann boundary condition data at each DoF support point.
     */
    Vector<double> neumann_bc;
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
     * Dirichlet boundary condition data at each DoF support point.
     */
    Vector<double> dirichlet_bc;
    Vector<double> dirichlet_bc_internal_dof_numbering;

    /**
     * Right hand side vector.
     */
    Vector<double> system_rhs;
    Vector<double> system_rhs_internal_dof_numbering;

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
     * Numerical solution for the Dirichlet domain.
     */
    Vector<double> solution_for_dirichlet_domain;
    Vector<double> solution_for_dirichlet_domain_internal_dof_numbering;
    /**
     * Numerical solution for the Neumann domain.
     */
    Vector<double> solution_for_neumann_domain;
    Vector<double> solution_for_neumann_domain_internal_dof_numbering;

    /**
     * Analytical solution for the Dirichlet domain.
     */
    Vector<double> analytical_solution_for_dirichlet_domain;
    /**
     * Analytical solution for the Neumann domain.
     */
    Vector<double> analytical_solution_for_neumann_domain;
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
    dof_handler_for_dirichlet_space_on_dirichlet_domain.clear();
    dof_handler_for_dirichlet_space_on_neumann_domain.clear();
    dof_handler_for_neumann_space_on_dirichlet_domain.clear();
    dof_handler_for_neumann_space_on_neumann_domain.clear();

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

    initialize_boundary_ids();

    extract_boundary_mesh();
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::initialize_boundary_ids()
  {}

  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::extract_boundary_mesh()
  {
    switch (problem_type)
      {
        case DirichletBCProblem:
          {
            /**
             * Extract the whole boundary mesh as the triangulation for the
             * Dirichlet domain.
             */
            map_from_dirichlet_boundary_mesh_to_volume_mesh =
              GridGenerator::extract_boundary_mesh(
                volume_triangulation, triangulation_for_dirichlet_domain);

            std::ofstream mesh_file("dirichlet_domain_mesh.msh");
            GridOut().write_msh(triangulation_for_dirichlet_domain, mesh_file);

            break;
          }
        case NeumannBCProblem:
          {
            /**
             * Extract the whole boundary mesh as the triangulation for the
             * Neumann domain.
             */
            map_from_neumann_boundary_mesh_to_volume_mesh =
              GridGenerator::extract_boundary_mesh(
                volume_triangulation, triangulation_for_neumann_domain);

            std::ofstream mesh_file("neumann_domain_mesh.msh");
            GridOut().write_msh(triangulation_for_neumann_domain, mesh_file);

            break;
          }
        case MixedBCProblem:
          {
            /**
             * Extract part of the boundary mesh as the triangulation for the
             * Dirichlet domain.
             */
            map_from_dirichlet_boundary_mesh_to_volume_mesh =
              GridGenerator::extract_boundary_mesh(
                volume_triangulation,
                triangulation_for_dirichlet_domain,
                boundary_ids_for_dirichlet_domain);

            /**
             * Extract part of the boundary mesh as the triangulation for the
             * Neumann domain.
             */
            map_from_neumann_boundary_mesh_to_volume_mesh =
              GridGenerator::extract_boundary_mesh(
                volume_triangulation,
                triangulation_for_neumann_domain,
                boundary_ids_for_neumann_domain);

            std::ofstream mesh_file("dirichlet_domain_mesh.msh");
            GridOut().write_msh(triangulation_for_dirichlet_domain, mesh_file);
            mesh_file.close();

            mesh_file.open("neumann_domain_mesh.msh");
            GridOut().write_msh(triangulation_for_neumann_domain, mesh_file);

            break;
          }
        default:
          {
            Assert(false, ExcInternalError());

            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::setup_system()
  {
    switch (problem_type)
      {
        case DirichletBCProblem:
          {
            dof_handler_for_dirichlet_space_on_dirichlet_domain.initialize(
              triangulation_for_dirichlet_domain, fe_for_dirichlet_space);
            dof_handler_for_neumann_space_on_dirichlet_domain.initialize(
              triangulation_for_dirichlet_domain, fe_for_neumann_space);

            const unsigned int n_dofs_for_dirichlet_space_on_dirichlet_domain =
              dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs();
            const unsigned int n_dofs_for_neumann_space_on_dirichlet_domain =
              dof_handler_for_neumann_space_on_dirichlet_domain.n_dofs();

            if (!is_use_hmat)
              {
                /**
                 * If full matrices are used for verification purpose,
                 * allocate memory for them here.
                 */
                V1_matrix.reinit(n_dofs_for_neumann_space_on_dirichlet_domain,
                                 n_dofs_for_neumann_space_on_dirichlet_domain);
                K2_matrix_with_mass_matrix.reinit(
                  n_dofs_for_neumann_space_on_dirichlet_domain,
                  n_dofs_for_dirichlet_space_on_dirichlet_domain);
              }
            else
              {
                /**
                 * Build the DoF-to-cell topology.
                 */
                build_dof_to_cell_topology(
                  dof_to_cell_topo_for_dirichlet_space_on_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain);
                build_dof_to_cell_topology(
                  dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
                  dof_handler_for_neumann_space_on_dirichlet_domain);

                /**
                 * Generate lists of DoF indices.
                 */
                dof_indices_for_dirichlet_space_on_dirichlet_domain.resize(
                  dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
                dof_indices_for_neumann_space_on_dirichlet_domain.resize(
                  dof_handler_for_neumann_space_on_dirichlet_domain.n_dofs());
                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_dirichlet_space_on_dirichlet_domain);
                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_neumann_space_on_dirichlet_domain);

                /**
                 * Get the spatial coordinates of the support points.
                 */
                support_points_for_dirichlet_space_on_dirichlet_domain.resize(
                  dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  support_points_for_dirichlet_space_on_dirichlet_domain);

                support_points_for_neumann_space_on_dirichlet_domain.resize(
                  dof_handler_for_neumann_space_on_dirichlet_domain.n_dofs());
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_dirichlet_domain,
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  support_points_for_neumann_space_on_dirichlet_domain);

                /**
                 * Calculate the average mesh cell size at each support point.
                 */
                dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain
                  .assign(dof_handler_for_dirichlet_space_on_dirichlet_domain
                            .n_dofs(),
                          0);
                map_dofs_to_average_cell_size(
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain);

                dof_average_cell_size_for_neumann_space_on_dirichlet_domain
                  .assign(
                    dof_handler_for_neumann_space_on_dirichlet_domain.n_dofs(),
                    0);
                map_dofs_to_average_cell_size(
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  dof_average_cell_size_for_neumann_space_on_dirichlet_domain);

                /**
                 * Initialize the cluster trees.
                 */
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

                /**
                 * Create the block cluster trees.
                 */
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

                /**
                 * Initialize \hmatrices.
                 */
                V1_hmat = HMatrixSymm<spacedim>(bct_for_bilinear_form_V1,
                                                max_hmat_rank);
                K2_hmat_with_mass_matrix =
                  HMatrix<spacedim>(bct_for_bilinear_form_K2,
                                    max_hmat_rank,
                                    HMatrixSupport::Property::general,
                                    HMatrixSupport::BlockType::diagonal_block);

                /**
                 * Initialize the preconditioner.
                 */
                V1_hmat_preconditioner = HMatrixSymmPreconditioner<spacedim>(
                  bct_for_bilinear_form_V1, max_hmat_rank_for_preconditioner);
              }

            /**
             * Interpolate the Dirichlet boundary data.
             */
            dirichlet_bc.reinit(n_dofs_for_dirichlet_space_on_dirichlet_domain);
            VectorTools::interpolate(
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              *dirichlet_bc_functor_ptr,
              dirichlet_bc);

            if (is_use_hmat)
              {
                /**
                 * Permute the Dirichlet boundary data by following the mapping
                 * from internal to external DoF numbering.
                 */
                dirichlet_bc_internal_dof_numbering.reinit(
                  n_dofs_for_dirichlet_space_on_dirichlet_domain);
                permute_vector(
                  dirichlet_bc,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  dirichlet_bc_internal_dof_numbering);
              }

            /**
             * Allocate memory for the right-hand-side vector and solution
             * vector.
             */
            system_rhs.reinit(n_dofs_for_neumann_space_on_dirichlet_domain);
            solution_for_dirichlet_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain);

            if (is_use_hmat)
              {
                system_rhs_internal_dof_numbering.reinit(
                  n_dofs_for_neumann_space_on_dirichlet_domain);
                solution_for_dirichlet_domain_internal_dof_numbering.reinit(
                  n_dofs_for_neumann_space_on_dirichlet_domain);
              }

            // DEBUG: export analytical solution for comparison.
            analytical_solution_for_dirichlet_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain);

            break;
          }
        case NeumannBCProblem:
          {
            dof_handler_for_dirichlet_space_on_neumann_domain.initialize(
              triangulation_for_neumann_domain, fe_for_dirichlet_space);
            dof_handler_for_neumann_space_on_neumann_domain.initialize(
              triangulation_for_neumann_domain, fe_for_neumann_space);

            const unsigned int n_dofs_for_dirichlet_space_on_neumann_domain =
              dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs();
            const unsigned int n_dofs_for_neumann_space_on_neumann_domain =
              dof_handler_for_neumann_space_on_neumann_domain.n_dofs();

            if (!is_use_hmat)
              {
                /**
                 * If full matrices are used for verification purpose,
                 * allocate memory for them here.
                 */
                D1_matrix.reinit(n_dofs_for_dirichlet_space_on_neumann_domain,
                                 n_dofs_for_dirichlet_space_on_neumann_domain);
                K_prime2_matrix_with_mass_matrix.reinit(
                  n_dofs_for_dirichlet_space_on_neumann_domain,
                  n_dofs_for_neumann_space_on_neumann_domain);

                /**
                 * SLP matrix for solving the natural density \f$w_{\rm eq}\f$.
                 */
                V1_matrix.reinit(n_dofs_for_neumann_space_on_neumann_domain,
                                 n_dofs_for_neumann_space_on_neumann_domain);
              }
            else
              {
                /**
                 * Build the DoF-to-cell topology.
                 */
                build_dof_to_cell_topology(
                  dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
                  dof_handler_for_dirichlet_space_on_neumann_domain);
                build_dof_to_cell_topology(
                  dof_to_cell_topo_for_neumann_space_on_neumann_domain,
                  dof_handler_for_neumann_space_on_neumann_domain);

                /**
                 * Generate lists of DoF indices.
                 */
                dof_indices_for_dirichlet_space_on_neumann_domain.resize(
                  dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs());
                dof_indices_for_neumann_space_on_neumann_domain.resize(
                  dof_handler_for_neumann_space_on_neumann_domain.n_dofs());
                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_dirichlet_space_on_neumann_domain);
                gen_linear_indices<vector_uta, types::global_dof_index>(
                  dof_indices_for_neumann_space_on_neumann_domain);

                /**
                 * Get the spatial coordinates of the support points.
                 */
                support_points_for_dirichlet_space_on_neumann_domain.resize(
                  dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs());
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_neumann_domain,
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  support_points_for_dirichlet_space_on_neumann_domain);

                support_points_for_neumann_space_on_neumann_domain.resize(
                  dof_handler_for_neumann_space_on_neumann_domain.n_dofs());
                DoFTools::map_dofs_to_support_points(
                  kx_mapping_for_neumann_domain,
                  dof_handler_for_neumann_space_on_neumann_domain,
                  support_points_for_neumann_space_on_neumann_domain);

                /**
                 * Calculate the average mesh cell size at each support point.
                 */
                dof_average_cell_size_for_dirichlet_space_on_neumann_domain
                  .assign(
                    dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs(),
                    0);
                map_dofs_to_average_cell_size(
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  dof_average_cell_size_for_dirichlet_space_on_neumann_domain);

                dof_average_cell_size_for_neumann_space_on_neumann_domain
                  .assign(
                    dof_handler_for_neumann_space_on_neumann_domain.n_dofs(),
                    0);
                map_dofs_to_average_cell_size(
                  dof_handler_for_neumann_space_on_neumann_domain,
                  dof_average_cell_size_for_neumann_space_on_neumann_domain);

                /**
                 * Initialize the cluster trees.
                 */
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

                /**
                 * Create the block cluster trees.
                 */
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

                /**
                 * Initialize \hmatrices.
                 */
                D1_hmat = HMatrixSymm<spacedim>(bct_for_bilinear_form_D1,
                                                max_hmat_rank);
                K_prime2_hmat_with_mass_matrix =
                  HMatrix<spacedim>(bct_for_bilinear_form_K_prime2,
                                    max_hmat_rank,
                                    HMatrixSupport::Property::general,
                                    HMatrixSupport::BlockType::diagonal_block);

                /**
                 * SLP matrix for solving the natural density \f$w_{\rm eq}\f$.
                 */
                V1_hmat = HMatrixSymm<spacedim>(bct_for_bilinear_form_V1,
                                                max_hmat_rank);

                /**
                 * Initialize the preconditioners.
                 */
                V1_hmat_preconditioner = HMatrixSymmPreconditioner<spacedim>(
                  bct_for_bilinear_form_V1, max_hmat_rank_for_preconditioner);
                D1_hmat_preconditioner = HMatrixSymmPreconditioner<spacedim>(
                  bct_for_bilinear_form_D1, max_hmat_rank_for_preconditioner);
              }

            /**
             * Interpolate the Neumann boundary data.
             */
            neumann_bc.reinit(n_dofs_for_neumann_space_on_neumann_domain);
            VectorTools::interpolate(
              dof_handler_for_neumann_space_on_neumann_domain,
              *neumann_bc_functor_ptr,
              neumann_bc);

            if (is_use_hmat)
              {
                /**
                 * Permute the Neumann boundary data by following the mapping
                 * from internal to external DoF numbering.
                 */
                neumann_bc_internal_dof_numbering.reinit(
                  n_dofs_for_neumann_space_on_neumann_domain);
                permute_vector(
                  neumann_bc,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  neumann_bc_internal_dof_numbering);
              }

            /**
             * Allocate memory for the natural density \f$w_{\rm eq}\in
             * H^{-1/2}(\Gamma)\f$ and its associated right hand side vector.
             */
            natural_density.reinit(n_dofs_for_neumann_space_on_neumann_domain);
            system_rhs_for_natural_density.reinit(
              n_dofs_for_neumann_space_on_neumann_domain);

            /**
             * Allocate memory for the product of mass matrix and the natural
             * density.
             */
            mass_vmult_weq.reinit(
              dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs());

            /**
             * Allocate memory for the right-hand-side vector and solution
             * vector.
             */
            system_rhs.reinit(n_dofs_for_dirichlet_space_on_neumann_domain);
            solution_for_neumann_domain.reinit(
              n_dofs_for_dirichlet_space_on_neumann_domain);

            if (is_use_hmat)
              {
                system_rhs_internal_dof_numbering.reinit(
                  n_dofs_for_dirichlet_space_on_neumann_domain);
                solution_for_neumann_domain_internal_dof_numbering.reinit(
                  n_dofs_for_dirichlet_space_on_neumann_domain);
              }

            // DEBUG: export analytical solution for comparison.
            analytical_solution_for_neumann_domain.reinit(
              n_dofs_for_dirichlet_space_on_neumann_domain);

            break;
          }
        case MixedBCProblem:
          {
            dof_handler_for_dirichlet_space_on_dirichlet_domain.initialize(
              triangulation_for_dirichlet_domain, fe_for_dirichlet_space);
            dof_handler_for_neumann_space_on_dirichlet_domain.initialize(
              triangulation_for_dirichlet_domain, fe_for_neumann_space);
            dof_handler_for_dirichlet_space_on_neumann_domain.initialize(
              triangulation_for_neumann_domain, fe_for_dirichlet_space);
            dof_handler_for_neumann_space_on_neumann_domain.initialize(
              triangulation_for_neumann_domain, fe_for_neumann_space);

            const unsigned int n_dofs_for_dirichlet_space_on_dirichlet_domain =
              dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs();
            const unsigned int n_dofs_for_neumann_space_on_dirichlet_domain =
              dof_handler_for_neumann_space_on_dirichlet_domain.n_dofs();
            const unsigned int n_dofs_for_dirichlet_space_on_neumann_domain =
              dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs();
            const unsigned int n_dofs_for_neumann_space_on_neumann_domain =
              dof_handler_for_neumann_space_on_neumann_domain.n_dofs();

            if (!is_use_hmat)
              {
                /**
                 * If full matrices are used for verification purpose,
                 * allocate memory for them.
                 */
                V1_matrix.reinit(n_dofs_for_neumann_space_on_dirichlet_domain,
                                 n_dofs_for_neumann_space_on_dirichlet_domain);
                K1_matrix.reinit(n_dofs_for_neumann_space_on_dirichlet_domain,
                                 n_dofs_for_dirichlet_space_on_neumann_domain);

                AssertDimension(V1_matrix.m(), K1_matrix.m());

                K_prime1_matrix.reinit(
                  n_dofs_for_dirichlet_space_on_neumann_domain,
                  n_dofs_for_neumann_space_on_dirichlet_domain);
                D1_matrix.reinit(n_dofs_for_dirichlet_space_on_neumann_domain,
                                 n_dofs_for_dirichlet_space_on_neumann_domain);

                AssertDimension(K_prime1_matrix.m(), D1_matrix.m());
                AssertDimension(V1_matrix.n(), K_prime1_matrix.n());
                AssertDimension(K1_matrix.n(), D1_matrix.n());

                K2_matrix_with_mass_matrix.reinit(
                  n_dofs_for_neumann_space_on_dirichlet_domain,
                  n_dofs_for_dirichlet_space_on_dirichlet_domain);
                V2_matrix.reinit(n_dofs_for_neumann_space_on_dirichlet_domain,
                                 n_dofs_for_neumann_space_on_neumann_domain);

                AssertDimension(K2_matrix_with_mass_matrix.m(), V2_matrix.m());
                AssertDimension(K2_matrix_with_mass_matrix.m(), K1_matrix.m());

                D2_matrix.reinit(
                  n_dofs_for_dirichlet_space_on_neumann_domain,
                  n_dofs_for_dirichlet_space_on_dirichlet_domain);
                K_prime2_matrix_with_mass_matrix.reinit(
                  n_dofs_for_dirichlet_space_on_neumann_domain,
                  n_dofs_for_neumann_space_on_neumann_domain);

                AssertDimension(D2_matrix.m(),
                                K_prime2_matrix_with_mass_matrix.m());
                AssertDimension(D2_matrix.m(), K_prime1_matrix.m());
                AssertDimension(K2_matrix_with_mass_matrix.n(), D2_matrix.n());
                AssertDimension(V2_matrix.n(),
                                K_prime2_matrix_with_mass_matrix.n());
              }
            else
              {
                /**
                 * TODO Setup for mixed boundary value problem solved by
                 * \hmatrix.
                 */
              }

            /**
             * Interpolate the Dirichlet boundary data.
             */
            dirichlet_bc.reinit(n_dofs_for_dirichlet_space_on_dirichlet_domain);
            VectorTools::interpolate(
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              *dirichlet_bc_functor_ptr,
              dirichlet_bc);

            /**
             * Interpolate the Neumann boundary data.
             */
            neumann_bc.reinit(n_dofs_for_neumann_space_on_neumann_domain);
            VectorTools::interpolate(
              dof_handler_for_neumann_space_on_neumann_domain,
              *neumann_bc_functor_ptr,
              neumann_bc);

            /**
             * Allocate memory for the right-hand-side vector and solution
             * vector.
             */
            system_rhs.reinit(n_dofs_for_neumann_space_on_dirichlet_domain +
                              n_dofs_for_dirichlet_space_on_neumann_domain);
            solution_for_dirichlet_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain +
              n_dofs_for_dirichlet_space_on_neumann_domain);

            break;
          }
        default:
          {
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
        case DirichletBCProblem:
          {
            /**
             * Assemble the FEM scaled mass matrix, which is stored into the
             * full matrix for \f$K_2\f$.
             *
             * \mynote{The polynomial order specified for the Gauss-Legendre
             * quadrature rule for FEM integration is accurate for the
             * integration of \f$2N-1\f$-th polynomial, where \f$N\f is the
             * number of quadrature points in 1D.}
             */
            std::cerr << "=== Assemble scaled mass matrix ===" << std::endl;

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
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K2_matrix_with_mass_matrix);
              }
            else
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  -0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K2_matrix_with_mass_matrix);
              }

            /**
             * Assemble the DLP matrix, which is added with the previous
             * scaled FEM mass matrix.
             */
            std::cerr << "=== Assemble DLP matrix ===" << std::endl;
            assemble_bem_full_matrix(
              double_layer_kernel,
              1.0,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K2_matrix_with_mass_matrix);

            /**
             * Assemble the SLP matrix.
             */
            std::cerr << "=== Assemble SLP matrix ===" << std::endl;
            assemble_bem_full_matrix(
              single_layer_kernel,
              1.0,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              V1_matrix);

            /**
             * Calculate the RHS vector.
             */
            K2_matrix_with_mass_matrix.vmult(system_rhs, dirichlet_bc);

            break;
          }
        case NeumannBCProblem:
          {
            std::cerr << "=== Assemble scaled mass matrix ===" << std::endl;

            /**
             * For the interior Laplace problem, \f$\frac{1}{2}I\f$ is
             * assembled, while for the exterior Laplace problem,
             * \f$-\frac{1}{2}I\f$ is assembled.
             */
            if (is_interior_problem)
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  dof_handler_for_neumann_space_on_neumann_domain,
                  0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K_prime2_matrix_with_mass_matrix);
              }
            else
              {
                assemble_fem_scaled_mass_matrix(
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  dof_handler_for_neumann_space_on_neumann_domain,
                  -0.5,
                  QGauss<2>(fe_for_dirichlet_space.degree + 1),
                  K_prime2_matrix_with_mass_matrix);
              }

            /**
             * Assemble the ADLP matrix, which is added with the
             * previous
             * scaled FEM mass matrix.
             */
            std::cerr << "=== Assemble ADLP matrix ===" << std::endl;
            assemble_bem_full_matrix(
              adjoint_double_layer_kernel,
              -1.0,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_neumann_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K_prime2_matrix_with_mass_matrix);

            /**
             * Assemble the matrix for the hyper singular operator, where the
             * regularization method is adopted.
             */
            std::cerr << "=== Assemble D matrix ===" << std::endl;

            assemble_bem_full_matrix(
              hyper_singular_kernel,
              1.0,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              D1_matrix);

            /**
             * Calculate the RHS vector.
             */
            K_prime2_matrix_with_mass_matrix.vmult(system_rhs, neumann_bc);

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
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_neumann_space_on_neumann_domain,
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
        case MixedBCProblem:
          {
            /**
             * Assemble the negated FEM scaled mass matrix \f$\mathscr{I}_1\f$,
             * which is stored into the full matrix for \f$K_2\f$.
             */
            assemble_fem_scaled_mass_matrix(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              -0.5,
              QGauss<2>(fe_for_dirichlet_space.degree + 1),
              K2_matrix_with_mass_matrix);

            /**
             * Assemble the negated DLP matrix, which is added with
             * \f$-\frac{1}{2}\mathscr{I}_1\f$.
             */
            assemble_bem_full_matrix(
              double_layer_kernel,
              -1.0,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K2_matrix_with_mass_matrix);

            /**
             * Assemble the FEM scaled mass matrix \f$\mathscr{I}_2\f$, which
             * is stored into the full matrix for \f$K_2'\f$.
             */
            assemble_fem_scaled_mass_matrix(
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_neumann_space_on_neumann_domain,
              0.5,
              QGauss<2>(fe_for_dirichlet_space.degree + 1),
              K_prime2_matrix_with_mass_matrix);

            /**
             * Assemble the negated ADLP matrix, which is added with
             * \f$\frac{1}{2}\mathscr{I}_2\f$.
             */
            assemble_bem_full_matrix(
              adjoint_double_layer_kernel,
              -1.0,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_neumann_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K_prime2_matrix_with_mass_matrix);

            /**
             * Assemble the negated SLP matrix \f$\mathscr{V}_1\f$.
             */
            assemble_bem_full_matrix(
              single_layer_kernel,
              -1.0,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              V1_matrix);

            /**
             * Assemble the DLP matrix \f$\mathscr{K}_1\f$.
             */
            assemble_bem_full_matrix(
              double_layer_kernel,
              1.0,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                DifferentTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K1_matrix);

            /**
             * Assemble the ADLP matrix \f$\mathscr{K}_1'\f$.
             */
            assemble_bem_full_matrix(
              adjoint_double_layer_kernel,
              1.0,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                DifferentTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              K_prime1_matrix);

            /**
             * Assemble the hyper singular matrix \f$\mathscr{D}_1\f$.
             */
            assemble_bem_full_matrix(
              hyper_singular_kernel,
              1.0,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              D1_matrix);

            /**
             * Assemble the SLP matrix \f$\mathscr{V}_2\f$.
             */
            assemble_bem_full_matrix(
              single_layer_kernel,
              1.0,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_neumann_space_on_neumann_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                DifferentTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              V2_matrix);

            /**
             * Assemble the negated hyper singular matrix \f$\mathscr{D}_2\f$.
             */
            assemble_bem_full_matrix(
              hyper_singular_kernel,
              -1.0,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                DifferentTriangulations,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              D2_matrix);

            break;
          }
        default:
          {
            Assert(false, ExcInternalError());

            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assemble_hmatrix_system()
  {
    MultithreadInfo::set_thread_limit(thread_num);

    /**
     * Define the @p ACAConfig object.
     */
    ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

    switch (problem_type)
      {
        case DirichletBCProblem:
          {
            if (is_interior_problem)
              {
                std::cerr << "=== Assemble sigma I + K" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K2_hmat_with_mass_matrix,
                  aca_config,
                  double_layer_kernel,
                  1.0,
                  0.5,
                  dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
                  dof_to_cell_topo_for_dirichlet_space_on_dirichlet_domain,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  kx_mapping_for_dirichlet_domain,
                  ky_mapping_for_dirichlet_domain,
                  *kx_mapping_data_for_dirichlet_domain,
                  *ky_mapping_data_for_dirichlet_domain,
                  map_from_dirichlet_boundary_mesh_to_volume_mesh,
                  map_from_dirichlet_boundary_mesh_to_volume_mesh,
                  IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);
              }
            else
              {
                std::cerr << "=== Assemble (sigma-1) I + K" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K2_hmat_with_mass_matrix,
                  aca_config,
                  double_layer_kernel,
                  1.0,
                  -0.5,
                  dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
                  dof_to_cell_topo_for_dirichlet_space_on_dirichlet_domain,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_dirichlet_domain,
                  kx_mapping_for_dirichlet_domain,
                  ky_mapping_for_dirichlet_domain,
                  *kx_mapping_data_for_dirichlet_domain,
                  *ky_mapping_data_for_dirichlet_domain,
                  map_from_dirichlet_boundary_mesh_to_volume_mesh,
                  map_from_dirichlet_boundary_mesh_to_volume_mesh,
                  IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);
              }

            std::cerr << "=== Assemble SLP matrix ===" << std::endl;

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              V1_hmat,
              aca_config,
              single_layer_kernel,
              1.0,
              dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
              dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            // DEBUG: Output hmatrices.
            std::ofstream out("V_bct.dat");
            V1_hmat.HMatrix<spacedim, double>::write_leaf_set_by_iteration(
              out, 1e-12);
            out.close();

            out.open("K_bct.dat");
            K2_hmat_with_mass_matrix
              .HMatrix<spacedim, double>::write_leaf_set_by_iteration(out,
                                                                      1e-12);
            out.close();

            out.open("matrices.dat");
            LAPACKFullMatrixExt<double> V_full, K_full;
            V1_hmat.HMatrix<spacedim, double>::convertToFullMatrix(V_full);
            K2_hmat_with_mass_matrix
              .HMatrix<spacedim, double>::convertToFullMatrix(K_full);

            V_full.print_formatted_to_mat(out, "V", 15, true, 25, "0");
            K_full.print_formatted_to_mat(out, "K", 15, true, 25, "0");
            out.close();

            /**
             * Calculate the RHS vector.
             */
            K2_hmat_with_mass_matrix.vmult(system_rhs_internal_dof_numbering,
                                           dirichlet_bc_internal_dof_numbering,
                                           HMatrixSupport::Property::general);

            break;
          }
        case NeumannBCProblem:
          {
            if (is_interior_problem)
              {
                std::cerr << "=== Assemble (1-sigma) I - K'" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K_prime2_hmat_with_mass_matrix,
                  aca_config,
                  adjoint_double_layer_kernel,
                  -1.0,
                  0.5,
                  dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
                  dof_to_cell_topo_for_neumann_space_on_neumann_domain,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  dof_handler_for_neumann_space_on_neumann_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  kx_mapping_for_neumann_domain,
                  ky_mapping_for_neumann_domain,
                  *kx_mapping_data_for_neumann_domain,
                  *ky_mapping_data_for_neumann_domain,
                  map_from_neumann_boundary_mesh_to_volume_mesh,
                  map_from_neumann_boundary_mesh_to_volume_mesh,
                  IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);
              }
            else
              {
                std::cerr << "=== Assemble -sigma I - K'" << std::endl;

                fill_hmatrix_with_aca_plus_smp(
                  thread_num,
                  K_prime2_hmat_with_mass_matrix,
                  aca_config,
                  adjoint_double_layer_kernel,
                  -1.0,
                  -0.5,
                  dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
                  dof_to_cell_topo_for_neumann_space_on_neumann_domain,
                  SauterQuadratureRule<dim>(5, 4, 4, 3),
                  QGauss<dim>(fe_order_for_dirichlet_space + 1),
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  dof_handler_for_neumann_space_on_neumann_domain,
                  *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
                  *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
                  kx_mapping_for_neumann_domain,
                  ky_mapping_for_neumann_domain,
                  *kx_mapping_data_for_neumann_domain,
                  *ky_mapping_data_for_neumann_domain,
                  map_from_neumann_boundary_mesh_to_volume_mesh,
                  map_from_neumann_boundary_mesh_to_volume_mesh,
                  IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                    SameTriangulations,
                  false);
              }

            /**
             * Calculate the RHS vector.
             */
            K_prime2_hmat_with_mass_matrix.vmult(
              system_rhs_internal_dof_numbering,
              neumann_bc_internal_dof_numbering,
              HMatrixSupport::Property::general);

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
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_neumann_space_on_neumann_domain,
              natural_density,
              QGauss<2>(fe_order_for_dirichlet_space + 1),
              mass_vmult_weq);

            /**
             * Assemble the regularized bilinear form for the hyper-singular
             * operator along with the stabilization term.
             */
            std::cerr << "=== Assemble D matrix ===" << std::endl;

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              D1_hmat,
              aca_config,
              hyper_singular_kernel,
              1.0,
              mass_vmult_weq,
              alpha_for_neumann,
              dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
              dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
              SauterQuadratureRule<dim>(5, 4, 4, 3),
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            break;
          }
        case MixedBCProblem:
          {
            break;
          }
        default:
          {
            Assert(false, ExcInternalError());

            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::assemble_hmatrix_preconditioner()
  {
    MultithreadInfo::set_thread_limit(thread_num);

    /**
     * Define the @p ACAConfig object.
     */
    ACAConfig aca_config(max_hmat_rank_for_preconditioner,
                         aca_relative_error_for_preconditioner,
                         eta_for_preconditioner);

    switch (problem_type)
      {
        case DirichletBCProblem:
          {
            std::cerr << "=== Assemble preconditioner for the SLP matrix ==="
                      << std::endl;

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              V1_hmat_preconditioner,
              aca_config,
              single_layer_kernel,
              1.0,
              dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
              dof_to_cell_topo_for_neumann_space_on_dirichlet_domain,
              SauterQuadratureRule<dim>(4, 3, 3, 2),
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              *dof_i2e_numbering_for_neumann_space_on_dirichlet_domain,
              kx_mapping_for_dirichlet_domain,
              ky_mapping_for_dirichlet_domain,
              *kx_mapping_data_for_dirichlet_domain,
              *ky_mapping_data_for_dirichlet_domain,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            /**
             * Perform Cholesky factorisation of the preconditioner.
             */
            V1_hmat_preconditioner.compute_cholesky_factorization(
              max_hmat_rank_for_preconditioner);

            break;
          }
        case NeumannBCProblem:
          {
            std::cerr << "=== Assemble preconditioner for the D matrix ==="
                      << std::endl;

            fill_hmatrix_with_aca_plus_smp(
              thread_num,
              D1_hmat_preconditioner,
              aca_config,
              hyper_singular_kernel,
              1.0,
              mass_vmult_weq,
              alpha_for_neumann,
              dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
              dof_to_cell_topo_for_dirichlet_space_on_neumann_domain,
              SauterQuadratureRule<dim>(4, 3, 3, 2),
              dof_handler_for_dirichlet_space_on_neumann_domain,
              dof_handler_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              *dof_i2e_numbering_for_dirichlet_space_on_neumann_domain,
              kx_mapping_for_neumann_domain,
              ky_mapping_for_neumann_domain,
              *kx_mapping_data_for_neumann_domain,
              *ky_mapping_data_for_neumann_domain,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameTriangulations,
              true);

            /**
             * Perform Cholesky factorisation of the preconditioner.
             */
            D1_hmat_preconditioner.compute_cholesky_factorization(
              max_hmat_rank_for_preconditioner);

            break;
          }
        case MixedBCProblem:
          {
            Assert(false, ExcNotImplemented());
            break;
          }
        default:
          {
            Assert(false, ExcInternalError());
            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::solve()
  {
    if (!is_use_hmat)
      {
        SolverControl solver_control(1000, 1e-12);
        SolverCG<>    solver(solver_control);

        switch (problem_type)
          {
            case DirichletBCProblem:
              {
                solver.solve(V1_matrix,
                             solution_for_dirichlet_domain,
                             system_rhs,
                             PreconditionIdentity());

                break;
              }
            case NeumannBCProblem:
              {
                solver.solve(D1_matrix,
                             solution_for_neumann_domain,
                             system_rhs,
                             PreconditionIdentity());

                break;
              }
            case MixedBCProblem:
              {
                // TODO Solve full matrix for mixed boundary value problem

                break;
              }
            default:
              {
                Assert(false, ExcInternalError());
              }
          }
      }
    else
      {
        SolverControl            solver_control(1000, 1e-6, true, true);
        SolverCG<Vector<double>> solver(solver_control);

        switch (problem_type)
          {
            case DirichletBCProblem:
              {
                solver.solve(
                  V1_hmat,
                  solution_for_dirichlet_domain_internal_dof_numbering,
                  system_rhs_internal_dof_numbering,
                  V1_hmat_preconditioner);

                /**
                 * Permute the solution vector by following the mapping
                 * from external to internal DoF numbering.
                 */
                permute_vector(
                  solution_for_dirichlet_domain_internal_dof_numbering,
                  *dof_e2i_numbering_for_neumann_space_on_dirichlet_domain,
                  solution_for_dirichlet_domain);

                break;
              }
            case NeumannBCProblem:
              {
                solver.solve(D1_hmat,
                             solution_for_neumann_domain_internal_dof_numbering,
                             system_rhs_internal_dof_numbering,
                             D1_hmat_preconditioner);

                /**
                 * Permute the solution vector by following the mapping
                 * from external to internal DoF numbering.
                 */
                permute_vector(
                  solution_for_neumann_domain_internal_dof_numbering,
                  *dof_e2i_numbering_for_dirichlet_space_on_neumann_domain,
                  solution_for_neumann_domain);

                break;
              }
            case MixedBCProblem:
              {
                // TODO Solve H-matrix for mixed boundary value problem

                break;
              }
            default:
              {
                Assert(false, ExcInternalError());
              }
          }
      }
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::output_results()
  {
    std::ofstream                           vtk_output;
    DataOut<dim, DoFHandler<dim, spacedim>> data_out;

    switch (problem_type)
      {
        case DirichletBCProblem:
          {
            // DEBUG: Interpolate the analytical solution.
            VectorTools::interpolate(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              *neumann_bc_functor_ptr,
              analytical_solution_for_dirichlet_domain);

            vtk_output.open("solution_for_dirichlet_bc.vtk",
                            std::ofstream::out);

            data_out.add_data_vector(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              solution_for_dirichlet_domain,
              "solution");

            // DEBUG: export analytical solution for comparison.
            data_out.add_data_vector(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              analytical_solution_for_dirichlet_domain,
              "analytical_solution");

            data_out.add_data_vector(
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              dirichlet_bc,
              "dirichlet_bc");

            print_vector_to_mat(std::cout,
                                "solution",
                                solution_for_dirichlet_domain,
                                false);

            break;
          }
        case NeumannBCProblem:
          {
            // DEBUG: Interpolate the analytical solution.
            VectorTools::interpolate(
              dof_handler_for_dirichlet_space_on_neumann_domain,
              *dirichlet_bc_functor_ptr,
              analytical_solution_for_neumann_domain);

            vtk_output.open("solution_for_neumann_bc.vtk", std::ofstream::out);

            data_out.add_data_vector(
              dof_handler_for_dirichlet_space_on_neumann_domain,
              solution_for_neumann_domain,
              "solution");

            // DEBUG: export analytical solution for comparison.
            data_out.add_data_vector(
              dof_handler_for_dirichlet_space_on_neumann_domain,
              analytical_solution_for_neumann_domain,
              "analytical_solution");

            data_out.add_data_vector(
              dof_handler_for_neumann_space_on_neumann_domain,
              neumann_bc,
              "neumann_bc");

            print_vector_to_mat(std::cout,
                                "solution",
                                solution_for_neumann_domain,
                                false);

            break;
          }
        case MixedBCProblem:
          {
            vtk_output.open("solution_for_mixed_bc.vtk", std::ofstream::out);

            data_out.add_data_vector(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              solution_for_dirichlet_domain,
              "solution_on_dirichlet_domain");
            data_out.add_data_vector(
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              dirichlet_bc,
              "dirichlet_bc");
            data_out.add_data_vector(
              dof_handler_for_dirichlet_space_on_neumann_domain,
              solution_for_neumann_domain,
              "solution_on_neumann_domain");
            data_out.add_data_vector(
              dof_handler_for_neumann_space_on_neumann_domain,
              neumann_bc,
              "neumann_bc");

            print_vector_to_mat(std::cout,
                                "solution_on_dirichlet_domain",
                                solution_for_dirichlet_domain,
                                false);
            print_vector_to_mat(std::cout,
                                "solution_on_neumann_domain",
                                solution_for_neumann_domain,
                                false);
            break;
          }
        default:
          {
            Assert(false, ExcInternalError());
          }
      }

    data_out.build_patches();
    data_out.write_vtk(vtk_output);
  }


  template <int dim, int spacedim>
  void
  LaplaceBEM<dim, spacedim>::output_potential_at_target_points()
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
        case DirichletBCProblem:
          {
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
                std::cerr << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  double_layer_kernel,
                  -1.0,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
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
                std::cerr << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  single_layer_kernel,
                  1.0,
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  solution_for_dirichlet_domain,
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
                std::cerr << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  double_layer_kernel,
                  1.0,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
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
                std::cerr << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  single_layer_kernel,
                  -1.0,
                  dof_handler_for_neumann_space_on_dirichlet_domain,
                  solution_for_dirichlet_domain,
                  false,
                  potential_grid_points,
                  potential_values);
              }

            break;
          }
        case NeumannBCProblem:
          {
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
                std::cerr << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  double_layer_kernel,
                  -1.0,
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  solution_for_neumann_domain,
                  false,
                  potential_grid_points,
                  potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cerr << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  single_layer_kernel,
                  1.0,
                  dof_handler_for_neumann_space_on_neumann_domain,
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
                std::cerr << "=== Evaluate DLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  double_layer_kernel,
                  1.0,
                  dof_handler_for_dirichlet_space_on_neumann_domain,
                  solution_for_neumann_domain,
                  false,
                  potential_grid_points,
                  potential_values);

                /**
                 * Evaluate the single layer potential, which is the single
                 * layer potential integral operator applied to the Neumann
                 * data. \f[ \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y)
                 * \intd s_y \f]
                 */
                std::cerr << "=== Evaluate SLP potential values ==="
                          << std::endl;
                evaluate_potential_at_points(
                  single_layer_kernel,
                  -1.0,
                  dof_handler_for_neumann_space_on_neumann_domain,
                  neumann_bc,
                  false,
                  potential_grid_points,
                  potential_values);
              }

            break;
          }
        case MixedBCProblem:
          {
            // TODO
            break;
          }
        default:
          {
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
    setup_system();

    if (!is_use_hmat)
      {
        assemble_full_matrix_system();
      }
    else
      {
        assemble_hmatrix_system();
        assemble_hmatrix_preconditioner();
      }

    solve();
    output_results();
    output_potential_at_target_points();

    // DEBUG
    if (problem_type == ProblemType::NeumannBCProblem)
      {
        verify_neumann_solution_in_space();
      }
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
    Vector<double> v(
      dof_handler_for_dirichlet_space_on_neumann_domain.n_dofs());
    assemble_fem_mass_matrix_vmult<dim, spacedim, double, Vector<double>>(
      dof_handler_for_dirichlet_space_on_neumann_domain,
      dof_handler_for_neumann_space_on_neumann_domain,
      natural_density,
      QGauss<2>(fe_order_for_dirichlet_space + 1),
      v);
    std::cout << "Analytical solution <gamma_0 u, weq>="
              << analytical_solution_for_neumann_domain * v << "\n";
    std::cout << "Numerical solution <gamma_0 u, weq>="
              << solution_for_neumann_domain * v << std::endl;
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
        std::cerr
          << "=== Assemble SLP matrix for solving the natural density === "
          << std::endl;

        assemble_bem_full_matrix(
          single_layer_kernel,
          1.0,
          dof_handler_for_neumann_space_on_neumann_domain,
          dof_handler_for_neumann_space_on_neumann_domain,
          kx_mapping_for_neumann_domain,
          ky_mapping_for_neumann_domain,
          *kx_mapping_data_for_neumann_domain,
          *ky_mapping_data_for_neumann_domain,
          map_from_neumann_boundary_mesh_to_volume_mesh,
          map_from_neumann_boundary_mesh_to_volume_mesh,
          IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
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

        std::cerr
          << "=== Assemble SLP matrix for solving the natural density ==="
          << std::endl;

        fill_hmatrix_with_aca_plus_smp(
          thread_num,
          V1_hmat,
          aca_config,
          single_layer_kernel,
          1.0,
          dof_to_cell_topo_for_neumann_space_on_neumann_domain,
          dof_to_cell_topo_for_neumann_space_on_neumann_domain,
          SauterQuadratureRule<dim>(5, 4, 4, 3),
          dof_handler_for_neumann_space_on_neumann_domain,
          dof_handler_for_neumann_space_on_neumann_domain,
          *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
          *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
          kx_mapping_for_neumann_domain,
          ky_mapping_for_neumann_domain,
          *kx_mapping_data_for_neumann_domain,
          *ky_mapping_data_for_neumann_domain,
          map_from_neumann_boundary_mesh_to_volume_mesh,
          map_from_neumann_boundary_mesh_to_volume_mesh,
          IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
            SameTriangulations,
          true);
      }

    /**
     * Assemble the RHS vector for solving the natural density \f$w_{\rm
     * eq}\f$.
     */
    std::cerr << "=== Assemble RHS vector for natural density ===" << std::endl;

    assemble_rhs_linear_form_vector(
      1.0,
      dof_handler_for_neumann_space_on_neumann_domain,
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
        /**
         * Define the @p ACAConfig object.
         */
        ACAConfig aca_config(max_hmat_rank_for_preconditioner,
                             aca_relative_error_for_preconditioner,
                             eta_for_preconditioner);

        std::cerr << "=== Assemble preconditioner for the SLP matrix ==="
                  << std::endl;

        fill_hmatrix_with_aca_plus_smp(
          thread_num,
          V1_hmat_preconditioner,
          aca_config,
          single_layer_kernel,
          1.0,
          dof_to_cell_topo_for_neumann_space_on_neumann_domain,
          dof_to_cell_topo_for_neumann_space_on_neumann_domain,
          SauterQuadratureRule<dim>(4, 3, 3, 2),
          dof_handler_for_neumann_space_on_neumann_domain,
          dof_handler_for_neumann_space_on_neumann_domain,
          *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
          *dof_i2e_numbering_for_neumann_space_on_neumann_domain,
          kx_mapping_for_neumann_domain,
          ky_mapping_for_neumann_domain,
          *kx_mapping_data_for_neumann_domain,
          *ky_mapping_data_for_neumann_domain,
          map_from_neumann_boundary_mesh_to_volume_mesh,
          map_from_neumann_boundary_mesh_to_volume_mesh,
          IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
            SameTriangulations,
          true);

        /**
         * Perform Cholesky factorisation of the preconditioner.
         */
        V1_hmat_preconditioner.compute_cholesky_factorization(
          max_hmat_rank_for_preconditioner);

        solver.solve(V1_hmat,
                     natural_density,
                     system_rhs_for_natural_density,
                     V1_hmat_preconditioner);
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
  }
} // namespace IdeoBEM

#endif /* INCLUDE_LAPLACE_BEM_H_ */
