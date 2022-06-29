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
#include <deal.II/fe/mapping_q_generic.h>

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

#include "bem_general.h"
#include "bem_tools.h"
#include "block_cluster_tree.h"
#include "cluster_tree.h"
#include "hmatrix.h"
#include "laplace_kernels.h"
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
      const DoFHandler<dim1, spacedim1> &     dof_handler_for_test_space,
      const DoFHandler<dim1, spacedim1> &     dof_handler_for_trial_space,
      const MappingQGeneric<dim1, spacedim1> &kx_mapping,
      const MappingQGeneric<dim1, spacedim1> &ky_mapping,
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
               unsigned int mapping_order,
               ProblemType  problem_type,
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
               unsigned int mapping_order,
               ProblemType  problem_type,
               unsigned int n_min_for_ct,
               unsigned int n_min_for_bct,
               double       eta,
               unsigned int max_hmat_rank,
               double       aca_relative_error,
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

    void
    solve();

    void
    output_results();

    void
    output_potential_at_target_points();

    void
    run();

  private:
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

    unsigned int                   mapping_order;
    MappingQGeneric<dim, spacedim> mapping;

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
    LaplaceKernel::HyperSingularKernel<3> hyper_singular_kernel;

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
    ClusterTree<spacedim> ct_for_neumann_space_on_dirichlet_domain;
    ClusterTree<spacedim> ct_for_neumann_space_on_neumann_domain;
    ClusterTree<spacedim> ct_for_dirichlet_space_on_dirichlet_domain;
    ClusterTree<spacedim> ct_for_dirichlet_space_on_neumann_domain;

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
    HMatrix<spacedim> V1_hmat;
    HMatrix<spacedim> K1_hmat;
    HMatrix<spacedim> K_prime1_hmat;
    HMatrix<spacedim> D1_hmat;
    HMatrix<spacedim> V2_hmat;
    HMatrix<spacedim> K2_hmat_with_mass_matrix;
    HMatrix<spacedim> K_prime2_hmat_with_mass_matrix;
    HMatrix<spacedim> D2_hmat;

    /**
     * The sequence of all DoF indices with the values \f$0, 1, \cdots\f$ for
     * different DoFHandlers.
     */
    std::vector<types::global_dof_index>
      dof_indices_for_neumann_space_on_dirichlet_domain;
    std::vector<types::global_dof_index>
      dof_indices_for_dirichlet_space_on_neumann_domain;
    std::vector<types::global_dof_index>
      dof_indices_for_dirichlet_space_on_dirichlet_domain;
    std::vector<types::global_dof_index>
      dof_indices_for_neumann_space_on_neumann_domain;

    /**
     * The list of all support points associated with @p dof_indices.
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
     * Estimated average cell size values associated with @p dof_indices.
     */
    std::vector<float>
      dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain;
    std::vector<float>
      dof_average_cell_size_for_dirichlet_space_on_neumann_domain;
    std::vector<float>
      dof_average_cell_size_for_neumann_space_on_dirichlet_domain;
    std::vector<float>
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
     * Pointer to the Neumann boundary condition function object.
     */
    Function<spacedim> *neumann_bc_functor_ptr;

    /**
     * Neumann boundary condition data at each DoF support point.
     */
    Vector<double> neumann_bc;

    /**
     * Pointer to the Dirichlet boundary condition function object.
     */
    Function<spacedim> *dirichlet_bc_functor_ptr;

    /**
     * Dirichlet boundary condition data at each DoF support point.
     */
    Vector<double> dirichlet_bc;

    /**
     * Right hand side vector.
     */
    Vector<double> system_rhs;

    Vector<double> solution_for_dirichlet_domain;
    Vector<double> solution_for_neumann_domain;
  };


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::LaplaceBEM()
    : fe_order_for_dirichlet_space(0)
    , fe_order_for_neumann_space(0)
    , problem_type(UndefinedProblem)
    , thread_num(0)
    , fe_for_dirichlet_space(0)
    , fe_for_neumann_space(0)
    , mapping_order(0)
    , mapping(0)
    , is_use_hmat(false)
    , n_min_for_ct(0)
    , n_min_for_bct(0) // By default, it is the same as the @p n_min_for_ct
    , eta(0)
    , max_hmat_rank(0)
    , aca_relative_error(0)
    , neumann_bc_functor_ptr(nullptr)
    , dirichlet_bc_functor_ptr(nullptr)
  {}


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::LaplaceBEM(
    unsigned int fe_order_for_dirichlet_space,
    unsigned int fe_order_for_neumann_space,
    unsigned int mapping_order,
    ProblemType  problem_type,
    unsigned int thread_num)
    : fe_order_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_order_for_neumann_space(fe_order_for_neumann_space)
    , problem_type(problem_type)
    , thread_num(thread_num)
    , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_for_neumann_space(fe_order_for_neumann_space)
    , mapping_order(mapping_order)
    , mapping(mapping_order)
    , is_use_hmat(false)
    , n_min_for_ct(0)
    , n_min_for_bct(0)
    , eta(0)
    , max_hmat_rank(0)
    , aca_relative_error(0)
    , neumann_bc_functor_ptr(nullptr)
    , dirichlet_bc_functor_ptr(nullptr)
  {}


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::LaplaceBEM(
    unsigned int fe_order_for_dirichlet_space,
    unsigned int fe_order_for_neumann_space,
    unsigned int mapping_order,
    ProblemType  problem_type,
    unsigned int n_min_for_ct,
    unsigned int n_min_for_bct,
    double       eta,
    unsigned int max_hmat_rank,
    double       aca_relative_error,
    unsigned int thread_num)
    : fe_order_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_order_for_neumann_space(fe_order_for_neumann_space)
    , problem_type(problem_type)
    , thread_num(thread_num)
    , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_for_neumann_space(fe_order_for_neumann_space)
    , mapping_order(mapping_order)
    , mapping(mapping_order)
    , is_use_hmat(true)
    , n_min_for_ct(n_min_for_ct)
    , n_min_for_bct(n_min_for_bct)
    , eta(eta)
    , max_hmat_rank(max_hmat_rank)
    , aca_relative_error(aca_relative_error)
    , neumann_bc_functor_ptr(nullptr)
    , dirichlet_bc_functor_ptr(nullptr)
  {}


  template <int dim, int spacedim>
  LaplaceBEM<dim, spacedim>::~LaplaceBEM()
  {
    dof_handler_for_dirichlet_space_on_dirichlet_domain.clear();
    dof_handler_for_dirichlet_space_on_neumann_domain.clear();
    dof_handler_for_neumann_space_on_dirichlet_domain.clear();
    dof_handler_for_neumann_space_on_neumann_domain.clear();

    neumann_bc_functor_ptr   = nullptr;
    dirichlet_bc_functor_ptr = nullptr;
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
                 * TODO Setup for Dirichlet problem solved by \hmatrix.
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
             * Allocate memory for the right-hand-side vector and solution
             * vector.
             */
            system_rhs.reinit(n_dofs_for_neumann_space_on_dirichlet_domain);
            solution_for_dirichlet_domain.reinit(
              n_dofs_for_neumann_space_on_dirichlet_domain);

            break;
          }
        case NeumannBCProblem:
          {
            // TODO
            Assert(false, ExcNotImplemented());

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
            assemble_fem_scaled_mass_matrix(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              dof_handler_for_dirichlet_space_on_dirichlet_domain,
              0.5,
              QGauss<2>(fe_for_dirichlet_space.degree + 1),
              K2_matrix_with_mass_matrix);

            // DEBUG
            print_matrix_to_mat(
              std::cout, "I", K2_matrix_with_mass_matrix, 15, false, 25, "0");

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
              mapping,
              mapping,
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
              mapping,
              mapping,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameDoFHandlers,
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
            // TODO Assemble full mass matrices for NeumannBCProblem. The
            // formulation of the representation formula and the boundary
            // integral equation should be determined.
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
              mapping,
              mapping,
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
              mapping,
              mapping,
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
              mapping,
              mapping,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              map_from_dirichlet_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameDoFHandlers,
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
              mapping,
              mapping,
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
              mapping,
              mapping,
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
              mapping,
              mapping,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              map_from_neumann_boundary_mesh_to_volume_mesh,
              IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                SameDoFHandlers,
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
              mapping,
              mapping,
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
              mapping,
              mapping,
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
                // TODO Solve full matrix for Neumann problem

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
        switch (problem_type)
          {
            case DirichletBCProblem:
              {
                // TODO Solve H-matrix for Dirichlet problem

                break;
              }
            case NeumannBCProblem:
              {
                // TODO Solve H-matrix for Neumann problem

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
            vtk_output.open("solution_for_dirichlet_bc.vtk",
                            std::ofstream::out);

            data_out.add_data_vector(
              dof_handler_for_neumann_space_on_dirichlet_domain,
              solution_for_dirichlet_domain,
              "solution");
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
            vtk_output.open("solution_for_neumann_bc.vtk", std::ofstream::out);

            data_out.add_data_vector(
              dof_handler_for_dirichlet_space_on_neumann_domain,
              solution_for_neumann_domain,
              "solution");
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

    /**
     * Evaluate the double layer potential, which is the negated double
     * layer potential integral operator applied to the Dirichlet data.
     * \f[
     * -\int_{\Gamma} \widetilde{\gamma}_{1,y} G(x,y) \gamma_0^{\rm
     * int} u(y) \intd s_y
     * \f]
     */
    std::cerr << "=== Evaluate DLP potential values ===" << std::endl;
    evaluate_potential_at_points(
      double_layer_kernel,
      -1.0,
      dof_handler_for_dirichlet_space_on_dirichlet_domain,
      dirichlet_bc,
      true,
      potential_grid_points,
      potential_values);

    /**
     * Evaluate the single layer potential, which is the single layer
     * potential integral operator applied to the Neumann data.
     * \f[
     * \int_{\Gamma} G(x,y) \widetilde{\gamma}_{1,y} u(y) \intd s_y
     * \f]
     */
    std::cerr << "=== Evaluate SLP potential values ===" << std::endl;
    evaluate_potential_at_points(
      single_layer_kernel,
      1.0,
      dof_handler_for_neumann_space_on_dirichlet_domain,
      solution_for_dirichlet_domain,
      true,
      potential_grid_points,
      potential_values);

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
        // TODO Assemble \hmatrix system
      }

    // DEBUG
    print_matrix_to_mat(std::cout, "V", V1_matrix, 15, false, 25, "0");
    print_matrix_to_mat(
      std::cout, "IK", K2_matrix_with_mass_matrix, 15, false, 25, "0");
    print_vector_to_mat(std::cout, "b", system_rhs, false);

    solve();
    output_results();
    output_potential_at_target_points();
  }
} // namespace IdeoBEM

#endif /* INCLUDE_LAPLACE_BEM_H_ */
