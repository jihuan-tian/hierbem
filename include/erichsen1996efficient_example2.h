/*
 * erichsen1996efficient-example2.h
 *
 *  Created on: 2020年11月27日
 *      Author: jihuan
 */

#ifndef INCLUDE_ERICHSEN1996EFFICIENT_EXAMPLE2_H_
#define INCLUDE_ERICHSEN1996EFFICIENT_EXAMPLE2_H_

#include <deal.II/base/function.h>
#include <deal.II/base/graph_coloring.h>

#include <deal.II/lac/vector.h>

// Basic quantities
#include <deal.II/base/point.h>
// Multithreading
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

// Triangulation
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// Numerical quadrature
#include <deal.II/base/quadrature_lib.h>

// DOFs manipulation
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// H1-conforming finite element shape functions.
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// Linear algebra related
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <boost/progress.hpp>

#include <fstream>
#include <functional>
#include <map>
#include <set>

#include "aca_plus.h"
#include "bem_general.h"
#include "bem_kernels.h"
#include "block_cluster_tree.h"
#include "cluster_tree.h"
#include "debug_tools.h"
#include "generic_functors.h"
#include "hmatrix.h"
#include "lapack_full_matrix_ext.h"
#include "laplace_bem.h"
#include "simple_bounding_box.h"
#include "unary_template_arg_containers.h"

//#define GRAPH_COLORING

using namespace dealii;

namespace IdeoBEM
{
  namespace Erichsen1996Efficient
  {
    template <int dim,
              int spacedim,
              typename RangeNumberType,
              typename MatrixType>
    friend void
    assemble_fem_scaled_mass_matrix(
      const DoFHandler<dim, spacedim> &dof_handler_for_test_space,
      const DoFHandler<dim, spacedim> &dof_handler_for_ansatz_space,
      const RangeNumberType            factor,
      const Quadrature<dim> &          quad_rule,
      MatrixType &                     target_full_matrix);

    template <int dim,
              int spacedim,
              typename RangeNumberType,
              typename MatrixType>
    friend void
    assemble_bem_full_matrix(
      const KernelFunction<spacedim, RangeNumberType> &kernel,
      const DoFHandler<dim, spacedim> &     dof_handler_for_test_space,
      const DoFHandler<dim, spacedim> &     dof_handler_for_ansatz_space,
      const MappingQGeneric<dim, spacedim> &kx_mapping,
      const MappingQGeneric<dim, spacedim> &ky_mapping,
      const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                     typename Triangulation<dim + 1, spacedim>::face_iterator>
        &map_from_test_space_mesh_to_volume_mesh,
      const std::map<typename Triangulation<dim, spacedim>::cell_iterator,
                     typename Triangulation<dim + 1, spacedim>::face_iterator>
        &map_from_ansatz_space_mesh_to_volume_mesh,
      const DetectCellNeighboringTypeMethod method_for_cell_neighboring_type,
      const SauterQuadratureRule &          sauter_quad_rule,
      MatrixType &                          target_full_matrix);

    class Example2
    {
    public:
      /**
       * Enum for the type of Laplace problem
       */
      enum ProblemType
      {
        NeumannBCProblem,   //!< NeumannBCProblem
        DirichletBCProblem, //!< DirichletBCProblem
        MixedBCProblem      //!< MixedBCProblem
      };

      /**
       * Function object for the Dirichlet boundary condition data, which is
       * also the solution of the Neumann problem. The analytical expression is:
       * \f[
       * u=\frac{1}{4\pi\norm{x-x_0}}
       * \f]
       */
      class DirichletBC : public Function<3>
      {
      public:
        // N.B. This function should be defined outside class NeumannBC or class
        // Example2, if no inline.
        DirichletBC()
          : Function<3>()
          , x0(0.25, 0.25, 0.25)
        {}

        DirichletBC(const Point<3> &x0)
          : Function<3>()
          , x0(x0)
        {}

        double
        value(const Point<3> &p, const unsigned int component = 0) const
        {
          (void)component;
          return 1.0 / 4.0 / numbers::PI / (p - x0).norm();
        }

      private:
        /**
         * Location of the Dirac point source \f$\delta(x-x_0)\f$.
         */
        Point<3> x0;
      };

      /**
       * Function object for the Neumann boundary condition data, which is also
       * the solution of the Dirichlet problem. The analytical expression is
       * \f[
       * \frac{\pdiff u}{\pdiff n}\Big\vert_{\Gamma} = \frac{\langle x-x_c,x-x_0
       * \rangle}{4\pi\norm{x-x_0}^3\rho}
       * \f]
       */
      class NeumannBC : public Function<3>
      {
      public:
        // N.B. This function should be defined outside class NeumannBC and
        // class Example2, if not inline.
        NeumannBC()
          : Function<3>()
          , x0(0.25, 0.25, 0.25)
          , model_sphere_center(0.0, 0.0, 0.0)
          , model_sphere_radius(1.0)
        {}

        NeumannBC(const Point<3> &x0, const Point<3> &center, double radius)
          : Function<3>()
          , x0(x0)
          , model_sphere_center(center)
          , model_sphere_radius(radius)
        {}

        double
        value(const Point<3> &p, const unsigned int component = 0) const
        {
          (void)component;

          Tensor<1, 3> diff_vector = p - x0;

          return ((p - model_sphere_center) * diff_vector) / 4.0 / numbers::PI /
                 Utilities::fixed_power<3>(diff_vector.norm()) /
                 model_sphere_radius;
        }

      private:
        Point<3> x0;
        Point<3> model_sphere_center;
        double   model_sphere_radius;
      };

      /**
       * Default constructor
       */
      Example2();

      /**
       * Constructor for full matrix builder and solver, which is only for
       * verification purpose.
       *
       * @param mesh_file_name
       * @param fe_order_for_dirichlet_space
       * @param fe_order_for_neumann_space
       * @param mapping_order
       * @param problem_type
       * @param thread_num
       */
      Example2(const std::string &mesh_file_name,
               unsigned int       fe_order_for_dirichlet_space = 1,
               unsigned int       fe_order_for_neumann_space   = 0,
               unsigned int       mapping_order                = 1,
               ProblemType        problem_type = DirichletBCProblem,
               unsigned int       thread_num   = 4);

      /**
       * Constructor for \hmatrix builder and solver.
       *
       * @param mesh_file_name
       * @param fe_order_for_dirichlet_space
       * @param fe_order_for_neumann_space
       * @param mapping_order
       * @param problem_type
       * @param thread_num
       * @param n_min_for_ct
       * @param n_min_for_bct
       * @param eta
       * @param max_hmat_rank
       * @param aca_relative_error
       */
      Example2();

      ~Example2();

      /**
       * Read the mesh from a file, which abandons the manifold description.
       */
      void
      read_mesh();

      /**
       * Extract the boundary mesh from the volume mesh for BEM.
       */
      void
      extract_boundary_mesh();

      void
      setup_system();

      /**
       * Assemble the system matrices as \hmatrices.
       */
      void
      assemble_system_as_hmatrices(
        const bool enable_build_symmetric_hmat = false);

      /**
       * Assemble the system matrices as \hmatrices (SMP version) without
       * incorporating FEM mass matrix multiplied by the factor 0.5.
       */
      void
      assemble_system_as_hmatrices_smp(
        const bool enable_build_symmetric_hmat = false);

      /**
       * Assemble the system matrices as \hmatrices (SMP version) incorporating
       * FEM mass matrix multiplied by the factor 0.5.
       */
      void
      assemble_system_as_hmatrices_with_mass_matrix_smp(
        const bool enable_build_symmetric_hmat = false);

      void
      run();

      void
      output_results();

      FullMatrix<double> &
      get_system_matrix();

      const FullMatrix<double> &
      get_system_matrix() const;

      FullMatrix<double> &
      get_system_rhs_matrix();

      const FullMatrix<double> &
      get_system_rhs_matrix() const;

      Vector<double> &
      get_system_rhs();

      const Vector<double> &
      get_system_rhs() const;

      ClusterTree<3> &
      get_ct();

      const ClusterTree<3> &
      get_ct() const;

      BlockClusterTree<3> &
      get_bct();

      const BlockClusterTree<3> &
      get_bct() const;

      std::vector<Point<3>> &
      get_all_support_points();

      const std::vector<Point<3>> &
      get_all_support_points() const;

      std::vector<types::global_dof_index> &
      get_dof_indices();

      const std::vector<types::global_dof_index> &
      get_dof_indices() const;

      HMatrix<3> &
      get_dlp_hmat();

      const HMatrix<3> &
      get_dlp_hmat() const;

      HMatrix<3> &
      get_slp_hmat();

      const HMatrix<3> &
      get_slp_hmat() const;

    private:
      /**
       * Generate the quadrangular surface mesh on the model sphere.
       *
       * @param number_of_refinements
       */
      void
      generate_mesh(unsigned int number_of_refinements = 0);

      /**
       * Calculate the neighboring type for each pair of cells.
       */
      void
      calc_cell_neighboring_types();

      void
      assemble_full_matrix_system_smp();

      /**
       * For debug purpose: Verify the function @p sauter_assemble_on_one_pair_of_dofs.
       */
      void
      assemble_system_via_pairs_of_dofs(bool is_assemble_fem_mat = true);

      /**
       * For debug purpose: Build SLP matrix only using SMP parallelization.
       */
      void
      assemble_slp_smp();

      void
      assemble_system_serial();

      void
      solve();

      /**
       * Mesh file to be read into the triangulation.
       */
      std::string mesh_file_name;
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
      Triangulation<3, 3> volume_triangulation;

      /**
       * Triangulation for the Dirichlet domain.
       */
      Triangulation<2, 3> triangulation_for_dirichlet_domain;

      /**
       * Triangulation for the Neumann domain.
       */
      Triangulation<2, 3> triangulation_for_neumann_domain;

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
      std::map<typename Triangulation<2, 3>::cell_iterator,
               typename Triangulation<3, 3>::face_iterator>
        map_from_dirichlet_boundary_mesh_to_volume_mesh;

      /**
       * Map from cell iterators in the surface mesh for the Neumann domain to
       * the face iterators in the original volume mesh.
       */
      std::map<typename Triangulation<2, 3>::cell_iterator,
               typename Triangulation<3, 3>::face_iterator>
        map_from_neumann_boundary_mesh_to_volume_mesh;

      /**
       * Finite element \f$H^{\frac{1}{2}+s}\f$ for the Dirichlet space. At
       * present, it is implemented as a continuous Lagrange space.
       */
      FE_Q<2, 3> fe_for_dirichlet_space;
      /**
       * Finite element \f$H^{-\frac{1}{2}+s}\f$ for the Neumann space. At
       * present, it is implemented as a discontinuous Lagrange space.
       */
      FE_DGQ<2, 3> fe_for_neumann_space;

      /**
       * Definition of DoFHandlers for a series of combination of finite element
       * spaces and triangulations.
       */
      DoFHandler<2, 3> dof_handler_for_dirichlet_space_on_dirichlet_domain;
      DoFHandler<2, 3> dof_handler_for_dirichlet_space_on_neumann_domain;
      DoFHandler<2, 3> dof_handler_for_neumann_space_on_dirichlet_domain;
      DoFHandler<2, 3> dof_handler_for_neumann_space_on_neumann_domain;

      unsigned int          mapping_order;
      MappingQGeneric<2, 3> mapping;

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

      // Location of the unit Dirac point source.
      Point<3> x0;
      Point<3> model_sphere_center;
      double   model_sphere_radius;

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

      bool is_use_hmat;

      /**
       * Cluster trees
       */
      ClusterTree<3> ct_for_neumann_space_on_dirichlet_domain;
      ClusterTree<3> ct_for_neumann_space_on_neumann_domain;
      ClusterTree<3> ct_for_dirichlet_space_on_dirichlet_domain;
      ClusterTree<3> ct_for_dirichlet_space_on_neumann_domain;

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
      BlockClusterTree<3> bct_for_bilinear_form_V1;
      BlockClusterTree<3> bct_for_bilinear_form_K1;
      BlockClusterTree<3> bct_for_bilinear_form_K_prime1;
      BlockClusterTree<3> bct_for_bilinear_form_D1;
      BlockClusterTree<3> bct_for_bilinear_form_V2;
      BlockClusterTree<3> bct_for_bilinear_form_K2;
      BlockClusterTree<3> bct_for_bilinear_form_K_prime2;
      BlockClusterTree<3> bct_for_bilinear_form_D2;

      /**
       * \hmatrices corresponding to discretized bilinear forms in the
       * mixed boundary value problem, which contain all possible cases.
       */
      HMatrix<3> V1_hmat;
      HMatrix<3> K1_hmat;
      HMatrix<3> K_prime1_hmat;
      HMatrix<3> D1_hmat;
      HMatrix<3> V2_hmat;
      HMatrix<3> K2_hmat_with_mass_matrix;
      HMatrix<3> K_prime2_hmat_with_mass_matrix;
      HMatrix<3> D2_hmat;

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
      std::vector<Point<3>>
        support_points_for_dirichlet_space_on_dirichlet_domain;
      std::vector<Point<3>>
        support_points_for_dirichlet_space_on_neumann_domain;
      std::vector<Point<3>>
                            support_points_for_neumann_space_on_dirichlet_domain;
      std::vector<Point<3>> support_points_for_neumann_space_on_neumann_domain;

      /**
       * Estimated average cell size values associated with @p dof_indices.
       */
      std::vector<double>
        dof_average_cell_size_for_dirichlet_space_on_dirichlet_domain;
      std::vector<double>
        dof_average_cell_size_for_dirichlet_space_on_neumann_domain;
      std::vector<double>
        dof_average_cell_size_for_neumann_space_on_dirichlet_domain;
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
       * Neumann boundary condition data at each DoF support point.
       */
      Vector<double> neumann_bc;

      /**
       * Dirichlet boundary condition data at each DoF support point.
       */
      Vector<double> dirichlet_bc;

      /**
       * Right hand side vector.
       */
      Vector<double> system_rhs;

      Vector<double> analytical_solution_for_dirichlet_domain;
      Vector<double> analytical_solution_for_neumann_domain;
      Vector<double> solution_for_dirichlet_domain;
      Vector<double> solution_for_neumann_domain;
    };

    Example2::Example2()
      : mesh_file_name("mesh.msh")
      , fe_order_for_dirichlet_space(1)
      , fe_order_for_neumann_space(0)
      , problem_type(DirichletBCProblem)
      , thread_num(4)
      , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
      , fe_for_neumann_space(fe_order_for_neumann_space)
      , mapping_order(1)
      , mapping(mapping_order)
      , x0(0.25, 0.25, 0.25)
      , model_sphere_center(0.0, 0.0, 0.0)
      , model_sphere_radius(1.0)
      , is_use_hmat(true)
      , n_min_for_ct(8)
      , n_min_for_bct(
          n_min_for_ct) // By default, it is the same as the @p n_min_for_ct
      , eta(1.0)
      , max_hmat_rank(2)
      , aca_relative_error(1e-2)
    {}


    Example2::Example2(const std::string &mesh_file_name,
                       unsigned int       fe_order_for_dirichlet_space,
                       unsigned int       fe_order_for_neumann_space,
                       unsigned int       mapping_order,
                       ProblemType        problem_type,
                       unsigned int       thread_num)
      : mesh_file_name(mesh_file_name)
      , fe_order_for_dirichlet_space(fe_order_for_dirichlet_space)
      , fe_order_for_neumann_space(fe_order_for_neumann_space)
      , problem_type(problem_type)
      , thread_num(thread_num)
      , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
      , fe_for_neumann_space(fe_order_for_neumann_space)
      , mapping_order(mapping_order)
      , mapping(mapping_order)
      , x0(0.25, 0.25, 0.25)
      , model_sphere_center(0.0, 0.0, 0.0)
      , model_sphere_radius(1.0)
      , is_use_hmat(false)
      , n_min_for_ct(8)
      , n_min_for_bct(n_min_for_bct)
      , eta(1.0)
      , max_hmat_rank(2)
      , aca_relative_error(1e-2)
    {}


    Example2::~Example2()
    {
      dof_handler_for_dirichlet_space_on_dirichlet_domain.clear();
      dof_handler_for_dirichlet_space_on_neumann_domain.clear();
      dof_handler_for_neumann_space_on_dirichlet_domain.clear();
      dof_handler_for_neumann_space_on_neumann_domain.clear();
    }

    void
    Example2::generate_mesh(unsigned int number_of_refinements)
    {
      // Generate the initial mesh.
      GridGenerator::hyper_ball((Triangulation<3>)volume_triangulation,
                                model_sphere_center,
                                model_sphere_radius,
                                true);

      // Output the initial mesh.
      GridOut       grid_out;
      std::string   base_name("sphere-");
      std::ofstream mesh_file(base_name + std::string("0.msh"));
      grid_out.write_msh(volume_triangulation, mesh_file);

      // Refine the mesh.
      for (unsigned int i = 0; i < number_of_refinements; i++)
        {
          volume_triangulation.refine_global(1);
          std::ofstream mesh_file(base_name + std::to_string(i + 1) +
                                  std::string(".msh"));
          grid_out.write_msh(volume_triangulation, mesh_file);
        }

      extract_boundary_mesh();
    }


    void
    Example2::read_mesh()
    {
      GridIn<3, 3> grid_in;
      grid_in.attach_triangulation(volume_triangulation);
      std::fstream mesh_file(mesh_file_name);
      grid_in.read_msh(mesh_file);

      // TODO Initialize the sets for boundary ids.

      extract_boundary_mesh();
    }


    void
    Example2::extract_boundary_mesh()
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

              break;
            }
        }
    }


    void
    Example2::calc_cell_neighboring_types()
    {
      const unsigned int n_active_cells =
        triangulation_for_dirichlet_domain.n_active_cells();
      FullMatrix<unsigned int> cell_neighboring_type_matrix(n_active_cells,
                                                            n_active_cells);

      types::global_vertex_index i = 0;
      for (const auto &first_cell :
           triangulation_for_dirichlet_domain.active_cell_iterators())
        {
          std::array<types::global_vertex_index,
                     GeometryInfo<2>::vertices_per_cell>
            first_cell_vertex_indices(get_vertex_indices<2, 3>(first_cell));

          types::global_vertex_index j = 0;
          for (const auto &second_cell :
               triangulation_for_dirichlet_domain.active_cell_iterators())
            {
              std::array<types::global_vertex_index,
                         GeometryInfo<2>::vertices_per_cell>
                second_cell_vertex_indices(
                  get_vertex_indices<2, 3>(second_cell));

              std::vector<types::global_vertex_index> vertex_index_intersection;
              vertex_index_intersection.reserve(
                GeometryInfo<2>::vertices_per_cell);
              cell_neighboring_type_matrix(i, j) =
                IdeoBEM::detect_cell_neighboring_type<2>(
                  first_cell_vertex_indices,
                  second_cell_vertex_indices,
                  vertex_index_intersection);

              j++;
            }

          i++;
        }

      deallog << "Cell neighboring types..." << std::endl;
      cell_neighboring_type_matrix.print(deallog);
      deallog << std::endl;
    }


    void
    Example2::setup_system()
    {
      switch (problem_type)
        {
          case DirichletBCProblem:
            {
              dof_handler_for_dirichlet_space_on_dirichlet_domain.initialize(
                triangulation_for_dirichlet_domain, fe_for_dirichlet_space);
              dof_handler_for_neumann_space_on_dirichlet_domain.initialize(
                triangulation_for_dirichlet_domain, fe_for_neumann_space);

              const unsigned int n_dofs_for_neumann_space_on_dirichlet_domain =
                dof_handler_for_neumann_space_on_dirichlet_domain.n_dofs();
              const unsigned int
                n_dofs_for_dirichlet_space_on_dirichlet_domain =
                  dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs();

              if (!is_use_hmat)
                {
                  /**
                   * If full matrices are used for verification purpose,
                   * allocate memory for them.
                   */
                  V1_matrix.reinit(
                    n_dofs_for_neumann_space_on_dirichlet_domain,
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
               * Function objects for analytical boundary data.
               */
              DirichletBC dirichlet_analytical_data(x0);
              NeumannBC   neumann_analytical_data(x0,
                                                model_sphere_center,
                                                model_sphere_radius);

              /**
               * Interpolate the analytical Dirichlet boundary data.
               */
              dirichlet_bc.reinit(
                n_dofs_for_dirichlet_space_on_dirichlet_domain);
              VectorTools::interpolate(
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dirichlet_analytical_data,
                dirichlet_bc);

              /**
               * Generate the analytical solution, i.e. the Neumann boundary
               * data.
               */
              analytical_solution_for_dirichlet_domain.reinit(
                n_dofs_for_neumann_space_on_dirichlet_domain);
              VectorTools::interpolate(
                dof_handler_for_neumann_space_on_dirichlet_domain,
                neumann_analytical_data,
                analytical_solution_for_dirichlet_domain);

              /**
               * Allocate memory for rhs and solution vectors.
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

              const unsigned int
                n_dofs_for_dirichlet_space_on_dirichlet_domain =
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
                  V1_matrix.reinit(
                    n_dofs_for_neumann_space_on_dirichlet_domain,
                    n_dofs_for_neumann_space_on_dirichlet_domain);
                  K1_matrix.reinit(
                    n_dofs_for_neumann_space_on_dirichlet_domain,
                    n_dofs_for_dirichlet_space_on_neumann_domain);

                  AssertDimension(V1_matrix.m(), K1_matrix.m());

                  K_prime1_matrix.reinit(
                    n_dofs_for_dirichlet_space_on_neumann_domain,
                    n_dofs_for_neumann_space_on_dirichlet_domain);
                  D1_matrix.reinit(
                    n_dofs_for_dirichlet_space_on_neumann_domain,
                    n_dofs_for_dirichlet_space_on_neumann_domain);

                  AssertDimension(K_prime1_matrix.m(), D1_matrix.m());
                  AssertDimension(V1_matrix.n(), K_prime1_matrix.n());
                  AssertDimension(K1_matrix.n(), D1_matrix.n());

                  K2_matrix_with_mass_matrix.reinit(
                    n_dofs_for_neumann_space_on_dirichlet_domain,
                    n_dofs_for_dirichlet_space_on_dirichlet_domain);
                  V2_matrix.reinit(n_dofs_for_neumann_space_on_dirichlet_domain,
                                   n_dofs_for_neumann_space_on_neumann_domain);

                  AssertDimension(K2_matrix_with_mass_matrix.m(),
                                  V2_matrix.m());
                  AssertDimension(K2_matrix_with_mass_matrix.m(),
                                  K1_matrix.m());

                  D2_matrix.reinit(
                    n_dofs_for_dirichlet_space_on_neumann_domain,
                    n_dofs_for_dirichlet_space_on_dirichlet_domain);
                  K_prime2_matrix_with_mass_matrix.reinit(
                    n_dofs_for_dirichlet_space_on_neumann_domain,
                    n_dofs_for_neumann_space_on_neumann_domain);

                  AssertDimension(D2_matrix.m(),
                                  K_prime2_matrix_with_mass_matrix.m());
                  AssertDimension(D2_matrix.m(), K_prime1_matrix.m());
                  AssertDimension(K2_matrix_with_mass_matrix.n(),
                                  D2_matrix.n());
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
               * Function objects for analytical boundary data.
               */
              DirichletBC dirichlet_analytical_data(x0);
              NeumannBC   neumann_analytical_data(x0,
                                                model_sphere_center,
                                                model_sphere_radius);

              /**
               * Interpolate the analytical Dirichlet boundary data.
               */
              dirichlet_bc.reinit(
                n_dofs_for_dirichlet_space_on_dirichlet_domain);
              VectorTools::interpolate(
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dirichlet_analytical_data,
                dirichlet_bc);

              /**
               * Interpolate the analytical Neumann boundary data.
               */
              neumann_bc.reinit(n_dofs_for_neumann_space_on_neumann_domain);
              VectorTools::interpolate(
                dof_handler_for_neumann_space_on_neumann_domain,
                neumann_analytical_data,
                neumann_bc);

              /**
               * Generate the analytical solution of Neumann data in the
               * Dirichlet domain.
               */
              analytical_solution_for_dirichlet_domain.reinit(
                n_dofs_for_neumann_space_on_dirichlet_domain);
              VectorTools::interpolate(
                dof_handler_for_neumann_space_on_dirichlet_domain,
                neumann_analytical_data,
                analytical_solution_for_dirichlet_domain);

              /**
               * Generate the analytical solution of Dirichlet data in the
               * Neumann domain.
               */
              analytical_solution_for_neumann_domain.reinit(
                n_dofs_for_dirichlet_space_on_neumann_domain);
              VectorTools::interpolate(
                dof_handler_for_dirichlet_space_on_neumann_domain,
                dirichlet_analytical_data,
                analytical_solution_for_neumann_domain);

              /**
               * Allocate memory for rhs and solution vectors.
               */
              system_rhs.reinit(n_dofs_for_neumann_space_on_dirichlet_domain +
                                n_dofs_for_dirichlet_space_on_neumann_domain);
              solution_for_dirichlet_domain.reinit(
                n_dofs_for_neumann_space_on_dirichlet_domain +
                n_dofs_for_dirichlet_space_on_neumann_domain);

              break;
            }
        }
    }


    void
    Example2::assemble_full_matrix_system_smp()
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
               * number of quadrature points.}
               */
              assemble_fem_scaled_mass_matrix(
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dof_handler_for_neumann_space_on_dirichlet_domain,
                0.5,
                QGauss<2>(fe_for_dirichlet_space.degree + 1),
                K2_matrix_with_mass_matrix);

              /**
               * Assemble the DLP matrix, which is added with the previous
               * scaled FEM mass matrix.
               */
              assemble_bem_full_matrix(
                double_layer_kernel,
                1.0,
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dof_handler_for_neumann_space_on_dirichlet_domain,
                mapping,
                mapping,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  SameTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
                K2_matrix_with_mass_matrix);

              /**
               * Assemble the SLP matrix.
               */
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
                SauterQuadratureRule(5, 4, 4, 3),
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
               * Assemble the FEM scaled mass matrix \f$\mathscr{I}_1\f$, which
               * is stored into the full matrix for \f$K_2\f$.
               */
              assemble_fem_scaled_mass_matrix(
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dof_handler_for_neumann_space_on_dirichlet_domain,
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
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dof_handler_for_neumann_space_on_dirichlet_domain,
                mapping,
                mapping,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  SameTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
                K2_matrix_with_mass_matrix);

              /**
               * Assemble the FEM scaled mass matrix \f$\mathscr{I}_2\f$, which
               * is stored into the full matrix for \f$K_2'\f$.
               */
              assemble_fem_scaled_mass_matrix(
                dof_handler_for_neumann_space_on_neumann_domain,
                dof_handler_for_dirichlet_space_on_neumann_domain,
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
                dof_handler_for_neumann_space_on_neumann_domain,
                dof_handler_for_dirichlet_space_on_neumann_domain,
                mapping,
                mapping,
                map_from_neumann_boundary_mesh_to_volume_mesh,
                map_from_neumann_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  SameTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
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
                SauterQuadratureRule(5, 4, 4, 3),
                V1_matrix);

              /**
               * Assemble the DLP matrix \f$\mathscr{K}_1\f$.
               */
              assemble_bem_full_matrix(
                double_layer_kernel,
                1.0,
                dof_handler_for_dirichlet_space_on_neumann_domain,
                dof_handler_for_neumann_space_on_dirichlet_domain,
                mapping,
                mapping,
                map_from_neumann_boundary_mesh_to_volume_mesh,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  DifferentTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
                K1_matrix);

              /**
               * Assemble the ADLP matrix \f$\mathscr{K}_1'\f$.
               */
              assemble_bem_full_matrix(
                adjoint_double_layer_kernel,
                1.0,
                dof_handler_for_neumann_space_on_dirichlet_domain,
                dof_handler_for_dirichlet_space_on_neumann_domain,
                mapping,
                mapping,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                map_from_neumann_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  DifferentTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
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
                SauterQuadratureRule(5, 4, 4, 3),
                D1_matrix);

              /**
               * Assemble the SLP matrix \f$\mathscr{V}_2\f$.
               */
              assemble_bem_full_matrix(
                single_layer_kernel,
                1.0,
                dof_handler_for_neumann_space_on_neumann_domain,
                dof_handler_for_neumann_space_on_dirichlet_domain,
                mapping,
                mapping,
                map_from_neumann_boundary_mesh_to_volume_mesh,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  DifferentTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
                V2_matrix);

              /**
               * Assemble the negated hyper singular matrix \f$\mathscr{D}_2\f$.
               */
              assemble_bem_full_matrix(
                hyper_singular_kernel,
                -1.0,
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dof_handler_for_dirichlet_space_on_neumann_domain,
                mapping,
                mapping,
                map_from_dirichlet_boundary_mesh_to_volume_mesh,
                map_from_neumann_boundary_mesh_to_volume_mesh,
                IdeoBEM::BEMTools::DetectCellNeighboringTypeMethod::
                  DifferentTriangulations,
                SauterQuadratureRule(5, 4, 4, 3),
                D2_matrix);

              break;
            }
        }
    }


    void
    Example2::assemble_system_via_pairs_of_dofs(bool is_assemble_fem_mat)
    {
      if (is_assemble_fem_mat)
        {
          // Generate normal Gauss-Legendre quadrature rule for FEM
          // integration.
          QGauss<2> quadrature_formula_2d(fe_for_dirichlet_space.degree + 1);

          WorkStream::run(
            cell_iterator_pointer_pairs_for_mass_matrix.begin(),
            cell_iterator_pointer_pairs_for_mass_matrix.end(),
            std::bind(&Example2::assemble_scaled_mass_matrix_on_one_cell,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&Example2::copy_cell_local_to_global,
                      this,
                      std::placeholders::_1),
            CellWiseScratchData(fe_for_dirichlet_space,
                                fe_for_neumann_space,
                                quadrature_formula_2d,
                                update_values | update_JxW_values),
            CellWisePerTaskData(fe_for_dirichlet_space, fe_for_neumann_space));
        }

      /**
       * Generate 4D Gauss-Legendre quadrature rules for various cell
       * neighboring types.
       */
      const unsigned int quad_order_for_same_panel    = 5;
      const unsigned int quad_order_for_common_edge   = 4;
      const unsigned int quad_order_for_common_vertex = 4;
      const unsigned int quad_order_for_regular       = 3;

      QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
      QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
      QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
      QGauss<4> quad_rule_for_regular(quad_order_for_regular);

      /**
       * Precalculate data tables for shape values at quadrature points.
       *
       * \mynote{Precalculate shape function values and their gradient values
       * at each quadrature point. N.B.
       * 1. The data tables for shape function values and their gradient
       * values should be calculated for both function space on \f$K_x\f$ and
       * function space on \f$K_y\f$.
       * 2. Being different from the integral in FEM, the integral in BEM
       * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
       * (except the regular cell neighboring type), each of which should be
       * evaluated at a different set of quadrature points in the unit cell
       * after coordinate transformation from the parametric space. Therefore,
       * a dimension with respect to \f$k_3\f$ term index should be added to
       * the data table compared to the usual FEValues and this brings about
       * the class @p BEMValues.}
       */
      BEMValues<2, 3> bem_values(fe_for_dirichlet_space,
                                 fe_for_dirichlet_space,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      // Initialize the progress display.
      boost::progress_display pd(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs() *
          dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs(),
        std::cerr);

      PairCellWiseScratchData scratch_data(fe_for_dirichlet_space,
                                           fe_for_dirichlet_space,
                                           bem_values);
      PairCellWisePerTaskData per_task_data(fe_for_dirichlet_space,
                                            fe_for_dirichlet_space);

      /**
       * Build the DoF-to-cell topology.
       */
      std::vector<std::vector<unsigned int>> dof_to_cell_topo;
      build_dof_to_cell_topology(
        dof_to_cell_topo, dof_handler_for_dirichlet_space_on_dirichlet_domain);

      for (types::global_dof_index i = 0;
           i < dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs();
           i++)
        {
          for (types::global_dof_index j = 0;
               j < dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs();
               j++)
            {
              K2_matrix_with_mass_matrix(i, j) +=
                sauter_assemble_on_one_pair_of_dofs(
                  scratch_data,
                  per_task_data,
                  double_layer_kernel,
                  i,
                  j,
                  dof_to_cell_topo,
                  bem_values,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  dof_handler_for_dirichlet_space_on_dirichlet_domain,
                  mapping,
                  mapping);
              V1_matrix(i, j) += sauter_assemble_on_one_pair_of_dofs(
                scratch_data,
                per_task_data,
                single_layer_kernel,
                i,
                j,
                dof_to_cell_topo,
                bem_values,
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                dof_handler_for_dirichlet_space_on_dirichlet_domain,
                mapping,
                mapping);

              ++pd;
            }
        }

      // Calculate the right-hand side vector.
      V1_matrix.vmult(system_rhs, neumann_bc);
    }


    void
    Example2::assemble_system_as_hmatrices(
      const bool enable_build_symmetric_hmat)
    {
      // Generate normal Gauss-Legendre quadrature rule for FEM integration.
      QGauss<2> quadrature_formula_2d(fe_for_dirichlet_space.degree + 1);

      WorkStream::run(
        cell_iterator_pointer_pairs_for_mass_matrix.begin(),
        cell_iterator_pointer_pairs_for_mass_matrix.end(),
        std::bind(&Example2::assemble_scaled_mass_matrix_on_one_cell,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3),
        std::bind(&Example2::copy_cell_local_to_global,
                  this,
                  std::placeholders::_1),
        CellWiseScratchData(fe_for_dirichlet_space,
                            fe_for_neumann_space,
                            quadrature_formula_2d,
                            update_values | update_JxW_values),
        CellWisePerTaskData(fe_for_dirichlet_space, fe_for_neumann_space));

      /**
       * Generate 4D Gauss-Legendre quadrature rules for various cell
       * neighboring types.
       */
      const unsigned int quad_order_for_same_panel    = 5;
      const unsigned int quad_order_for_common_edge   = 4;
      const unsigned int quad_order_for_common_vertex = 4;
      const unsigned int quad_order_for_regular       = 3;

      QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
      QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
      QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
      QGauss<4> quad_rule_for_regular(quad_order_for_regular);

      /**
       * Precalculate data tables for shape values at quadrature points.
       *
       * \mynote{Precalculate shape function values and their gradient values
       * at each quadrature point. N.B.
       * 1. The data tables for shape function values and their gradient
       * values should be calculated for both function space on \f$K_x\f$ and
       * function space on \f$K_y\f$.
       * 2. Being different from the integral in FEM, the integral in BEM
       * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
       * (except the regular cell neighboring type), each of which should be
       * evaluated at a different set of quadrature points in the unit cell
       * after coordinate transformation from the parametric space. Therefore,
       * a dimension with respect to \f$k_3\f$ term index should be added to
       * the data table compared to the usual FEValues and this brings about
       * the class @p BEMValues.}
       */
      BEMValues<2, 3> bem_values(fe_for_dirichlet_space,
                                 fe_for_dirichlet_space,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      PairCellWiseScratchData scratch_data(fe_for_dirichlet_space,
                                           fe_for_dirichlet_space,
                                           bem_values);
      PairCellWisePerTaskData per_task_data(fe_for_dirichlet_space,
                                            fe_for_dirichlet_space);

      /**
       * Build the DoF-to-cell topology.
       */
      std::vector<std::vector<unsigned int>> dof_to_cell_topo;
      build_dof_to_cell_topology(
        dof_to_cell_topo, dof_handler_for_dirichlet_space_on_dirichlet_domain);

      /**
       * Generate a list of all DoF indices.
       */
      std::vector<types::global_dof_index> dof_indices(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
      gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

      /**
       * Get the spatial coordinates of the support points associated with DoF
       * indices.
       */
      std::vector<Point<3>> all_support_points(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
      DoFTools::map_dofs_to_support_points(
        mapping,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        all_support_points);

      /**
       * Calculate the average mesh cell size at each support point.
       */
      std::vector<double> dof_average_cell_size(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs(), 0);
      map_dofs_to_average_cell_size(
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_average_cell_size);

      /**
       * Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
       * indices.
       */
      ct_for_neumann_space_on_dirichlet_domain = ClusterTree<3>(
        dof_indices, all_support_points, dof_average_cell_size, n_min_for_ct);

      /**
       * Partition the cluster tree.
       */
      ct_for_neumann_space_on_dirichlet_domain.partition(all_support_points,
                                                         dof_average_cell_size);

      /**
       * Create the block cluster tree.
       */
      bct_for_bilinear_form_V1 =
        BlockClusterTree<3>(ct_for_neumann_space_on_dirichlet_domain,
                            ct_for_neumann_space_on_dirichlet_domain,
                            eta,
                            n_min_for_bct);

      /**
       * Perform admissible partition on the block cluster tree.
       */
      bct_for_bilinear_form_V1.partition(all_support_points);

      /**
       * Initialize the SLP and DLP \hmatrices.
       *
       * \comment{既然自己已经为 @p ClusterTree, @p BlockClusterTree 以及 @p HMatrix 定义了
       * 浅拷贝构造与赋值函数，那么自己就要在实际中大胆地使用。一开始不熟悉、不放心，多次使用且经过实践的验证就习以为常了。}
       */
      V1_hmat =
        HMatrix<3>(bct_for_bilinear_form_V,
                   max_hmat_rank,
                   (enable_build_symmetric_hmat ? HMatrixSupport::symmetric :
                                                  HMatrixSupport::general),
                   HMatrixSupport::diagonal_block);
      K2_hmat_with_mass_matrix =
        HMatrix<3>(bct_for_bilinear_form_V, max_hmat_rank);

      /**
       * Define the @p ACAConfig object.
       */
      ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

      /**
       * Fill the \hmatrices using ACA+ approximation.
       */
      fill_hmatrix_with_aca_plus(
        V1_hmat,
        scratch_data,
        per_task_data,
        aca_config,
        single_layer_kernel,
        dof_to_cell_topo,
        bem_values,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        mapping,
        mapping,
        enable_build_symmetric_hmat);

      fill_hmatrix_with_aca_plus(
        K2_hmat_with_mass_matrix,
        scratch_data,
        per_task_data,
        aca_config,
        double_layer_kernel,
        dof_to_cell_topo,
        bem_values,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        mapping,
        mapping,
        enable_build_symmetric_hmat);
    }


    void
    Example2::assemble_system_as_hmatrices_smp(
      const bool enable_build_symmetric_hmat)
    {
      // Generate normal Gauss-Legendre quadrature rule for FEM integration.
      QGauss<2> quadrature_formula_2d(fe_for_dirichlet_space.degree + 1);

      /**
       * Generate 4D Gauss-Legendre quadrature rules for various cell
       * neighboring types.
       */
      const unsigned int quad_order_for_same_panel    = 5;
      const unsigned int quad_order_for_common_edge   = 4;
      const unsigned int quad_order_for_common_vertex = 4;
      const unsigned int quad_order_for_regular       = 3;

      QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
      QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
      QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
      QGauss<4> quad_rule_for_regular(quad_order_for_regular);

      /**
       * Precalculate data tables for shape values at quadrature points.
       *
       * \mynote{Precalculate shape function values and their gradient values
       * at each quadrature point. N.B.
       * 1. The data tables for shape function values and their gradient
       * values should be calculated for both function space on \f$K_x\f$ and
       * function space on \f$K_y\f$.
       * 2. Being different from the integral in FEM, the integral in BEM
       * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
       * (except the regular cell neighboring type), each of which should be
       * evaluated at a different set of quadrature points in the unit cell
       * after coordinate transformation from the parametric space. Therefore,
       * a dimension with respect to \f$k_3\f$ term index should be added to
       * the data table compared to the usual FEValues and this brings about
       * the class @p BEMValues.}
       */
      BEMValues<2, 3> bem_values(fe_for_dirichlet_space,
                                 fe_for_dirichlet_space,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      /**
       * Build the DoF-to-cell topology.
       */
      std::vector<std::vector<unsigned int>> dof_to_cell_topo;
      build_dof_to_cell_topology(
        dof_to_cell_topo, dof_handler_for_dirichlet_space_on_dirichlet_domain);

      /**
       * Generate a list of all DoF indices.
       */
      std::vector<types::global_dof_index> dof_indices(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
      gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

      /**
       * Get the spatial coordinates of the support points associated with DoF
       * indices.
       */
      std::vector<Point<3>> all_support_points(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
      DoFTools::map_dofs_to_support_points(
        mapping,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        all_support_points);

      /**
       * Calculate the average mesh cell size at each support point.
       */
      std::vector<double> dof_average_cell_size(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs(), 0);
      map_dofs_to_average_cell_size(
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_average_cell_size);

      /**
       * Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
       * indices.
       */
      ct_for_neumann_space_on_dirichlet_domain = ClusterTree<3>(
        dof_indices, all_support_points, dof_average_cell_size, n_min_for_ct);

      /**
       * Partition the cluster tree.
       */
      ct_for_neumann_space_on_dirichlet_domain.partition(all_support_points,
                                                         dof_average_cell_size);

      /**
       * Create the block cluster tree.
       */
      bct_for_bilinear_form_V =
        BlockClusterTree<3>(ct_for_neumann_space_on_dirichlet_domain,
                            ct_for_neumann_space_on_dirichlet_domain,
                            eta,
                            n_min_for_bct);

      /**
       * Perform admissible partition on the block cluster tree.
       */
      bct_for_bilinear_form_V.partition(all_support_points);

      /**
       * Initialize the SLP and DLP \hmatrices.
       *
       * \comment{既然自己已经为 @p ClusterTree, @p BlockClusterTree 以及 @p HMatrix 定义了
       * 浅拷贝构造与赋值函数，那么自己就要在实际中大胆地使用。一开始不熟悉、不放心，多次使用且经过实践的验证就习以为常了。}
       */
      V1_hmat =
        HMatrix<3>(bct_for_bilinear_form_V,
                   max_hmat_rank,
                   (enable_build_symmetric_hmat ? HMatrixSupport::symmetric :
                                                  HMatrixSupport::general),
                   HMatrixSupport::diagonal_block);
      K2_hmat_with_mass_matrix =
        HMatrix<3>(bct_for_bilinear_form_V, max_hmat_rank);

      /**
       * Define the @p ACAConfig object.
       */
      ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

      /**
       * Fill the \hmatrices using ACA+ approximation.
       */
      //      fill_hmatrix_with_aca_plus_smp(thread_num,
      //                                     dlp_hmat,
      //                                     aca_config,
      //                                     dlp,
      //                                     dof_to_cell_topo,
      //                                     bem_values,
      //                                     dof_handler,
      //                                     dof_handler,
      //                                     mapping,
      //                                     mapping);
      //
      //      fill_hmatrix_with_aca_plus_smp(thread_num,
      //                                     slp_hmat,
      //                                     aca_config,
      //                                     slp,
      //                                     dof_to_cell_topo,
      //                                     bem_values,
      //                                     dof_handler,
      //                                     dof_handler,
      //                                     mapping,
      //                                     mapping);

      std::vector<HMatrix<3, double> *> hmats{&K2_hmat_with_mass_matrix,
                                              &V1_hmat};
      std::vector<KernelFunction<3> *>  kernels{&double_layer_kernel,
                                               &single_layer_kernel};

      fill_hmatrix_with_aca_plus_smp(
        thread_num,
        hmats,
        aca_config,
        kernels,
        dof_to_cell_topo,
        bem_values,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        mapping,
        mapping,
        enable_build_symmetric_hmat);
    }


    void
    Example2::assemble_system_as_hmatrices_with_mass_matrix_smp(
      const bool enable_build_symmetric_hmat)
    {
      /**
       * Generate 4D Gauss-Legendre quadrature rules for various cell
       * neighboring types.
       */
      const unsigned int quad_order_for_same_panel    = 5;
      const unsigned int quad_order_for_common_edge   = 4;
      const unsigned int quad_order_for_common_vertex = 4;
      const unsigned int quad_order_for_regular       = 3;

      QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
      QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
      QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
      QGauss<4> quad_rule_for_regular(quad_order_for_regular);

      /**
       * Precalculate data tables for shape values at quadrature points.
       *
       * \mynote{Precalculate shape function values and their gradient values
       * at each quadrature point. N.B.
       * 1. The data tables for shape function values and their gradient
       * values should be calculated for both function space on \f$K_x\f$ and
       * function space on \f$K_y\f$.
       * 2. Being different from the integral in FEM, the integral in BEM
       * handled by Sauter's quadrature rule has multiple parts of \f$k_3\f$
       * (except the regular cell neighboring type), each of which should be
       * evaluated at a different set of quadrature points in the unit cell
       * after coordinate transformation from the parametric space. Therefore,
       * a dimension with respect to \f$k_3\f$ term index should be added to
       * the data table compared to the usual FEValues and this brings about
       * the class @p BEMValues.}
       */
      BEMValues<2, 3> bem_values(fe_for_dirichlet_space,
                                 fe_for_dirichlet_space,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      /**
       * Build the DoF-to-cell topology.
       */
      std::vector<std::vector<unsigned int>> dof_to_cell_topo;
      build_dof_to_cell_topology(
        dof_to_cell_topo, dof_handler_for_dirichlet_space_on_dirichlet_domain);

      /**
       * Generate a list of all DoF indices.
       */
      std::vector<types::global_dof_index> dof_indices(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
      gen_linear_indices<vector_uta, types::global_dof_index>(dof_indices);

      /**
       * Get the spatial coordinates of the support points associated with DoF
       * indices.
       */
      std::vector<Point<3>> all_support_points(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs());
      DoFTools::map_dofs_to_support_points(
        mapping,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        all_support_points);

      /**
       * Calculate the average mesh cell size at each support point.
       */
      std::vector<double> dof_average_cell_size(
        dof_handler_for_dirichlet_space_on_dirichlet_domain.n_dofs(), 0);
      map_dofs_to_average_cell_size(
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_average_cell_size);

      /**
       * Initialize the cluster tree \f$T(I)\f$ and \f$T(J)\f$ for all the DoF
       * indices.
       */
      ct_for_neumann_space_on_dirichlet_domain = ClusterTree<3>(
        dof_indices, all_support_points, dof_average_cell_size, n_min_for_ct);

      /**
       * Partition the cluster tree.
       */
      ct_for_neumann_space_on_dirichlet_domain.partition(all_support_points,
                                                         dof_average_cell_size);

      /**
       * Create the block cluster tree.
       */
      bct_for_bilinear_form_V =
        BlockClusterTree<3>(ct_for_neumann_space_on_dirichlet_domain,
                            ct_for_neumann_space_on_dirichlet_domain,
                            eta,
                            n_min_for_bct);

      /**
       * Perform admissible partition on the block cluster tree.
       */
      bct_for_bilinear_form_V.partition(all_support_points);

      /**
       * Initialize the SLP and DLP \hmatrices.
       *
       * \comment{既然自己已经为 @p ClusterTree, @p BlockClusterTree 以及 @p HMatrix 定义了
       * 浅拷贝构造与赋值函数，那么自己就要在实际中大胆地使用。一开始不熟悉、不放心，多次使用且经过实践的验证就习以为常了。}
       */
      V1_hmat =
        HMatrix<3>(bct_for_bilinear_form_V,
                   max_hmat_rank,
                   (enable_build_symmetric_hmat ? HMatrixSupport::symmetric :
                                                  HMatrixSupport::general),
                   HMatrixSupport::diagonal_block);
      K2_hmat_with_mass_matrix =
        HMatrix<3>(bct_for_bilinear_form_V, max_hmat_rank);

      /**
       * Define the @p ACAConfig object.
       */
      ACAConfig aca_config(max_hmat_rank, aca_relative_error, eta);

      /**
       * Fill the \hmatrices using ACA+ approximation.
       */
      std::vector<HMatrix<3, double> *> hmats{&K2_hmat_with_mass_matrix,
                                              &V1_hmat};
      std::vector<KernelFunction<3> *>  kernels{&double_layer_kernel,
                                               &single_layer_kernel};

      /**
       * Generate normal Gauss-Legendre quadrature rule for FEM integration.
       */
      QGauss<2> quadrature_formula_2d(fe_for_dirichlet_space.degree + 1);

      /**
       * Factors before the mass matrix which is to be added into the DLP
       * \hmatrix. For Example 2 in the Erichsen1996Efficient paper, the
       * system matrix to be solved is \f$\frac{1}{2}I+K\f$. Therefore, the
       * mass matrix scaled by 0.5 is appended to the \hmatrix associated with
       * the DLP kernel function.
       */
      std::vector<double> mass_matrix_factors{0.5, 0};

      fill_hmatrix_with_aca_plus_smp(
        thread_num,
        hmats,
        mass_matrix_factors,
        aca_config,
        kernels,
        dof_to_cell_topo,
        bem_values,
        quadrature_formula_2d,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        dof_handler_for_dirichlet_space_on_dirichlet_domain,
        mapping,
        mapping,
        enable_build_symmetric_hmat);

      /**
       * Calculate the RHS vector.
       */
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              V1_hmat.vmult(system_rhs, neumann_bc, V1_hmat.get_property());

              break;
            }
          case DirichletBCProblem:
            {
              K2_hmat_with_mass_matrix.vmult(system_rhs, dirichlet_bc);

              break;
            }
        }
    }


    void
    Example2::assemble_slp_smp()
    {
      MultithreadInfo::set_thread_limit(4);

      // Precalculate shape function values and their gradient
      // values at each quadrature point. N.B.
      // 1. The data tables for shape function values and their gradient
      // values should be calculated for both function space on \f$K_x\f$ and
      // function space on \f$K_y\f$.
      // 2. Being different from the integral in FEM, the integral in BEM
      // handled by Sauter's quadrature rule has multiple parts of $k_3$
      // (except the regular cell neighboring type), each of which should be
      // evaluated at a different set of quadrature points in the unit cell
      // after coordinate transformation from the parametric space. Therefore,
      // a dimension with respect to $k_3$ term index should be added to the
      // data table compared to the usual FEValues.

      // Generate 4D Gauss-Legendre quadrature rules for various cell
      // neighboring types.
      const unsigned int quad_order_for_same_panel    = 5;
      const unsigned int quad_order_for_common_edge   = 4;
      const unsigned int quad_order_for_common_vertex = 4;
      const unsigned int quad_order_for_regular       = 4;

      QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
      QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
      QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
      QGauss<4> quad_rule_for_regular(quad_order_for_regular);

      // Precalculate data tables for shape values at quadrature points.
      BEMValues<2, 3> bem_values(fe_for_dirichlet_space,
                                 fe_for_dirichlet_space,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      // Initialize the progress display.
      boost::progress_display pd(
        triangulation_for_dirichlet_domain.n_active_cells(), std::cerr);

      PairCellWiseScratchData scratch_data(fe_for_dirichlet_space,
                                           fe_for_dirichlet_space,
                                           bem_values);
      PairCellWisePerTaskData per_task_data(fe_for_dirichlet_space,
                                            fe_for_dirichlet_space);

      // Calculate the term \f$(v, V(u))\f$.
      for (const auto &e : dof_handler_for_dirichlet_space_on_dirichlet_domain
                             .active_cell_iterators())
        {
#ifdef GRAPH_COLORING
          WorkStream::run(
            colored_cells,
            std::bind(&Example2::assemble_for_full_matrix_on_one_pair_of_cells,
                      this,
                      e,
                      std::placeholders::_1,
                      bem_values,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&Example2::copy_pair_of_cells_local_to_global,
                      this,
                      std::placeholders::_1),
            scratch_data,
            per_task_data);
#else
          WorkStream::run(
            dof_handler_for_dirichlet_space_on_dirichlet_domain.begin_active(),
            dof_handler_for_dirichlet_space_on_dirichlet_domain.end(),
            std::bind(&Example2::assemble_on_one_pair_of_cells_for_slp,
                      this,
                      e,
                      std::placeholders::_1,
                      bem_values,
                      std::placeholders::_2,
                      std::placeholders::_3),
            std::bind(&Example2::copy_pair_of_cells_local_to_global_for_slp,
                      this,
                      std::placeholders::_1),
            scratch_data,
            per_task_data);
#endif

          ++pd;
        }

      // Calculate the right-hand side vector.
      V1_matrix.vmult(system_rhs, neumann_bc);
    }


    void
    Example2::assemble_system_serial()
    {
      // Generate normal Gauss-Legendre quadrature rule for FEM integration.
      QGauss<2> quadrature_formula_2d(fe_for_dirichlet_space.degree + 1);

      const unsigned int dofs_per_cell = fe_for_dirichlet_space.dofs_per_cell;

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

      // Calculate the term $(v, \frac{1}{2}u)$.
      FEValues<2, 3> fe_values(fe_for_dirichlet_space,
                               quadrature_formula_2d,
                               update_values | update_JxW_values);

      // The memory of this vector should be allocated before the calling of
      // <code>get_dof_indices</code>.
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const unsigned int n_q_points = quadrature_formula_2d.size();

      // Loop over each active cell.
      for (const auto &cell :
           dof_handler_for_dirichlet_space_on_dirichlet_domain
             .active_cell_iterators())
        {
          // Clear the local matrix elements for a new cell.
          local_matrix = 0.;

          fe_values.reinit(cell);

          // Loop over each quadrature point in the current cell.
          for (unsigned int q = 0; q < n_q_points; q++)
            {
              // Iterate over test function DoFs.
              for (unsigned int i = 0; i < dofs_per_cell; i++)
                {
                  // Iterate over ansatz function DoFs.
                  for (unsigned int j = 0; j < dofs_per_cell; j++)
                    {
                      local_matrix(i, j) += 0.5 * fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) *
                                            fe_values.JxW(q);
                    }
                }
            }

          // Extract the DoF indices.
          cell->get_dof_indices(local_dof_indices);

          // Assemble the local matrix to system matrix.
          for (unsigned int i = 0; i < dofs_per_cell; i++)
            {
              for (unsigned int j = 0; j < dofs_per_cell; j++)
                {
                  K2_matrix_with_mass_matrix.add(local_dof_indices[i],
                                                 local_dof_indices[j],
                                                 local_matrix(i, j));
                }
            }
        }

      /**
       * Precalculate shape function values and their gradient values at each
       * quadrature point. N.B.
       *
       * 1. The data tables for shape function values and their gradient
       * values should be calculated for both function space on \f$K_x\f$ and
       * function space on \f$K_y\f$.
       *
       * 2. Being different from the integral in FEM, the integral in
       * Galerkin-BEM handled by Sauter's quadrature rule has multiple parts
       * of \f$k_3\f$ (except the regular cell neighboring type), each of
       * which should be evaluated at a different set of quadrature points in
       * the unit cell after coordinate transformation from the parametric
       * space. Therefore, a dimension with respect to \f$k_3\f$ term index
       * should be
       * added to the data table compared to the usual @p FEValues.
       */

      // Generate 4D Gauss-Legendre quadrature rules for various cell
      // neighboring types.
      const unsigned int quad_order_for_same_panel    = 5;
      const unsigned int quad_order_for_common_edge   = 4;
      const unsigned int quad_order_for_common_vertex = 4;
      const unsigned int quad_order_for_regular       = 3;

      QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
      QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
      QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
      QGauss<4> quad_rule_for_regular(quad_order_for_regular);

      // Precalculate data tables for shape values at quadrature points.
      BEMValues<2, 3> bem_values(fe_for_dirichlet_space,
                                 fe_for_dirichlet_space,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      // Initialize the progress display.
      boost::progress_display pd(
        triangulation_for_dirichlet_domain.n_active_cells() *
          triangulation_for_dirichlet_domain.n_active_cells(),
        deallog.get_console());

      // Calculate the term $(v, Ku)$ to be assembled into system matrix and
      // $(v, V(\psi))$ to be assembled into right-hand side vector.
      for (const auto &e : dof_handler_for_dirichlet_space_on_dirichlet_domain
                             .active_cell_iterators())
        {
          for (const auto &f :
               dof_handler_for_dirichlet_space_on_dirichlet_domain
                 .active_cell_iterators())
            {
              SauterQuadRule(K2_matrix_with_mass_matrix,
                             this->double_layer_kernel,
                             bem_values,
                             e,
                             f,
                             this->mapping,
                             this->mapping);

              SauterQuadRule(V1_matrix,
                             this->single_layer_kernel,
                             bem_values,
                             e,
                             f,
                             this->mapping,
                             this->mapping);

              ++pd;
            }
        }

      // Calculate the right-hand side vector.
      V1_matrix.vmult(system_rhs, neumann_bc);
    }


    void
    Example2::solve()
    {
      // Solve the system matrix using CG.
      SolverControl solver_control(1000, 1e-12);
      SolverCG<>    solver(solver_control);

      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              solver.solve(K2_matrix_with_mass_matrix,
                           solution_for_dirichlet_domain,
                           system_rhs,
                           PreconditionIdentity());

              break;
            }
          case DirichletBCProblem:
            {
              solver.solve(V1_matrix,
                           solution_for_dirichlet_domain,
                           system_rhs,
                           PreconditionIdentity());

              break;
            }
        }
    }


    void
    Example2::output_results()
    {
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              print_vector_to_mat(std::cout,
                                  "analytical_solution",
                                  analytical_solution_for_neumann_domain,
                                  false);

              break;
            }
          case DirichletBCProblem:
            {
              print_vector_to_mat(std::cout,
                                  "analytical_solution",
                                  analytical_solution_for_dirichlet_domain,
                                  false);

              break;
            }
        }

      print_vector_to_mat(std::cout,
                          "numerical_solution",
                          solution_for_dirichlet_domain,
                          false);

      DataOut<2, DoFHandler<2, 3>> data_out;
      data_out.attach_dof_handler(
        dof_handler_for_dirichlet_space_on_dirichlet_domain);
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              data_out.add_data_vector(analytical_solution_for_neumann_domain,
                                       "analytical_solution");
              data_out.add_data_vector(neumann_bc, "neumann_bc");

              break;
            }
          case DirichletBCProblem:
            {
              data_out.add_data_vector(analytical_solution_for_dirichlet_domain,
                                       "analytical_solution");
              data_out.add_data_vector(dirichlet_bc, "dirichlet_bc");

              break;
            }
        }

      data_out.add_data_vector(solution_for_dirichlet_domain,
                               "numerical_solution");
      data_out.build_patches();

      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              std::ofstream output("solution-neumann.vtk");
              data_out.write_vtk(output);

              break;
            }
          case DirichletBCProblem:
            {
              std::ofstream output("solution-dirichlet.vtk");
              data_out.write_vtk(output);

              break;
            }
        }
    }


    FullMatrix<double> &
    Example2::get_system_matrix()
    {
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              return K2_matrix_with_mass_matrix;

              break;
            }
          case DirichletBCProblem:
            {
              return V1_matrix;

              break;
            }
          default:
            {
              return K2_matrix_with_mass_matrix;
            }
        }
    }


    const FullMatrix<double> &
    Example2::get_system_matrix() const
    {
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              return K2_matrix_with_mass_matrix;

              break;
            }
          case DirichletBCProblem:
            {
              return V1_matrix;

              break;
            }
          default:
            {
              return K2_matrix_with_mass_matrix;
            }
        }
    }


    FullMatrix<double> &
    Example2::get_system_rhs_matrix()
    {
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              return V1_matrix;

              break;
            }
          case DirichletBCProblem:
            {
              return K2_matrix_with_mass_matrix;

              break;
            }
          default:
            {
              return V1_matrix;
            }
        }
    }


    const FullMatrix<double> &
    Example2::get_system_rhs_matrix() const
    {
      switch (problem_type)
        {
          case NeumannBCProblem:
            {
              return V1_matrix;

              break;
            }
          case DirichletBCProblem:
            {
              return K2_matrix_with_mass_matrix;

              break;
            }
          default:
            {
              return V1_matrix;
            }
        }
    }


    Vector<double> &
    Example2::get_system_rhs()
    {
      return system_rhs;
    }


    const Vector<double> &
    Example2::get_system_rhs() const
    {
      return system_rhs;
    }


    ClusterTree<3> &
    Example2::get_ct()
    {
      return ct_for_neumann_space_on_dirichlet_domain;
    }

    BlockClusterTree<3> &
    Example2::get_bct()
    {
      return bct_for_bilinear_form_V;
    }


    std::vector<Point<3>> &
    Example2::get_all_support_points()
    {
      return support_points_for_dirichlet_space_on_dirichlet_domain;
    }


    const std::vector<Point<3>> &
    Example2::get_all_support_points() const
    {
      return support_points_for_dirichlet_space_on_dirichlet_domain;
    }

    const BlockClusterTree<3> &
    Example2::get_bct() const
    {
      return bct_for_bilinear_form_V;
    }

    const ClusterTree<3> &
    Example2::get_ct() const
    {
      return ct_for_neumann_space_on_dirichlet_domain;
    }


    HMatrix<3> &
    Example2::get_dlp_hmat()
    {
      return K2_hmat_with_mass_matrix;
    }


    const HMatrix<3> &
    Example2::get_dlp_hmat() const
    {
      return K2_hmat_with_mass_matrix;
    }


    std::vector<types::global_dof_index> &
    Example2::get_dof_indices()
    {
      return dof_indices_for_neumann_space_on_dirichlet_domain;
    }


    const std::vector<types::global_dof_index> &
    Example2::get_dof_indices() const
    {
      return dof_indices_for_neumann_space_on_dirichlet_domain;
    }


    HMatrix<3> &
    Example2::get_slp_hmat()
    {
      return V1_hmat;
    }


    const HMatrix<3> &
    Example2::get_slp_hmat() const
    {
      return V1_hmat;
    }


    void
    Example2::run()
    {
      read_mesh();
      setup_system();

      if (!is_use_hmat)
        {
          if (thread_num > 1)
            {
              assemble_full_matrix_system_smp();

              // DEBUG: print out the assembled matrices.
              // std::ofstream out("matrices-assemble-on-cell-pair.dat");
              //          std::ofstream out(
              //            "matrices-assemble-on-cell-pair-with-mass-matrix.dat");
              //          print_matrix_to_mat(
              //            out, "dlp_cell_pair", dlp_matrix, 25, false, 15,
              //            "0");
              //          print_matrix_to_mat(
              //            out, "slp_cell_pair", slp_matrix, 25, false, 15,
              //            "0");
              //          out.close();
            }
          else
            {
              // assemble_system_serial();

              // For verification of the function @p assemble_system_via_pairs_of_dofs.
              // assemble_system_via_pairs_of_dofs();

              // For verification of the function @p fill_hmatrix_with_aca_plus in
              // @p aca_plus.h, during which the FEM matrix will not be assembled,
              // and only SLP and DLP full matrices are generated.
              assemble_system_via_pairs_of_dofs(false);

              // DEBUG: print out the assembled matrices.
              std::ofstream out("matrices-assemble-on-dof-pair.dat");
              print_matrix_to_mat(out,
                                  "dlp_dof_pair",
                                  K2_matrix_with_mass_matrix,
                                  25,
                                  false,
                                  15,
                                  "0");
              print_matrix_to_mat(
                out, "slp_dof_pair", V1_matrix, 25, false, 15, "0");
              out.close();
            }

          solve();
          output_results();
        }
      else
        {
          assemble_system_as_hmatrices_with_mass_matrix_smp(true);

          switch (problem_type)
            {
              case NeumannBCProblem:
                {
                  /**
                   * Perform the LU factorization of the system matrix
                   * \f$\frac{1}{2}I+K\f$.
                   */
                  K2_hmat_with_mass_matrix.compute_lu_factorization(
                    max_hmat_rank);

                  /**
                   * Solve the system equation using the direct LU solver.
                   */
                  K2_hmat_with_mass_matrix.solve_lu(
                    this->solution_for_dirichlet_domain, this->system_rhs);

                  break;
                }
              case DirichletBCProblem:
                {
                  /**
                   * Perform the Cholesky factorization of the system matrix
                   * \f$V\f$.
                   */
                  V1_hmat.compute_cholesky_factorization(max_hmat_rank);

                  /**
                   * Solve the system equation.
                   */
                  V1_hmat.solve_cholesky(this->solution_for_dirichlet_domain,
                                         this->system_rhs);

                  break;
                }
            }

          output_results();
        }
    }
  } // namespace Erichsen1996Efficient
} // namespace IdeoBEM

#endif /* INCLUDE_ERICHSEN1996EFFICIENT_EXAMPLE2_H_ */
