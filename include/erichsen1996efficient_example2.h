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
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// Linear algebra related
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <boost/progress.hpp>
#include <laplace_bem.h>

#include <fstream>
#include <functional>


#define RUN_SMP_PARALLEL
//#define GRAPH_COLORING

using namespace dealii;

namespace LaplaceBEM
{
  namespace Erichsen1996Efficient
  {
    class Example2
    {
    public:
      // Function object for the analytical solution.
      class AnalyticalSolution : public Function<3>
      {
      public:
        // N.B. This function should be defined outside class NeumannBC or class
        // Example2.
        AnalyticalSolution()
          : Function<3>()
          , x0(0.25, 0.25, 0.25)
        {}

        AnalyticalSolution(const Point<3> &x0)
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
         * Location of the point source.
         */
        Point<3> x0;
      };

      // Function object for the Neumann boundary condition data.
      class NeumannBC : public Function<3>
      {
      public:
        // N.B. This function should be defined outside class NeumannBC and
        // class Example2.
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

          return (model_sphere_center - p) * (-diff_vector) / 4.0 /
                 numbers::PI / std::pow(diff_vector.norm(), 3);
        }

      private:
        Point<3> x0;
        Point<3> model_sphere_center;
        double   model_sphere_radius;
      };

      Example2();
      Example2(const std::string &mesh_file_name, unsigned int fe_order = 2);
      ~Example2();

      void
      run();

      void
      output_results();

    private:
      std::string  mesh_file_name;
      unsigned int fe_order;

      // Generate the quadrangular surface mesh on the model sphere.
      void
      generate_mesh(unsigned int number_of_refinements = 0);

      /**
       * Read the mesh from a file, which abandons the manifold description.
       */
      void
      read_mesh();


      /**
       * Calculate the neighboring type for each pair of cells.
       */
      void
      calc_cell_neighboring_types();

      void
      setup_system();

      void
      assemble_on_one_cell(
        const typename DoFHandler<2, 3>::active_cell_iterator &cell_iter,
        CellWiseScratchData &                                  scratch,
        CellWisePerTaskData &                                  data);

      void
      copy_cell_local_to_global(const CellWisePerTaskData &data);

      /**
       * Assemble BEM matrices on a pair of cells, i.e. $K_x$ as the field cell
       * and $K_y$ as the source cell.
       * @param kx_cell_iter
       * @param ky_cell_iter
       * @param scratch
       * @param data
       */
      void
      assemble_on_one_pair_of_cells(
        const typename DoFHandler<2, 3>::active_cell_iterator &kx_cell_iter,
        const typename DoFHandler<2, 3>::active_cell_iterator &ky_cell_iter,
        const BEMValues<2, 3> &                                bem_values,
        PairCellWiseScratchData &                              scratch,
        PairCellWisePerTaskData &                              data);

      void
      copy_pair_of_cells_local_to_global(const PairCellWisePerTaskData &data);

      void
      assemble_system_smp();

      void
      assemble_system_serial();

      void
      solve();

      Triangulation<2, 3>   triangulation;
      FE_Q<2, 3>            fe;
      DoFHandler<2, 3>      dof_handler;
      MappingQGeneric<2, 3> mapping;

      // Generate the single layer kernel function object.
      LaplaceKernel::SingleLayerKernel<3> slp;
      // Generate the double layer kernel function object.
      LaplaceKernel::DoubleLayerKernel<3> dlp;

      // Location of the unit Dirac point source.
      Point<3> x0;
      Point<3> model_sphere_center;
      double   model_sphere_radius;

      /**
       * System matrix obtained from $(v, \frac{1}{2}u) + (v, Ku)$.
       * The first integral term in the sum is carried on each cell, while the
       * second integral term is carried out on each pair of cells.
       */
      FullMatrix<double> system_matrix;
      /**
       * The right hand side matrix obtained from $(v, Vu)$.
       */
      FullMatrix<double> system_rhs_matrix;
      /**
       * Neumann boundary condition data at each DoF support point.
       */
      Vector<double> neumann_bc;
      /**
       * Right hand side vector for the problem obtained from the product of
       * <code>system_rhs_matrix</code> and <code>neumann_bc</code>
       */
      Vector<double> system_rhs;

      Vector<double> analytical_solution;
      Vector<double> solution;
    };

    Example2::Example2()
      : mesh_file_name("mesh.msh")
      , fe_order(1)
      , fe(fe_order)
      , dof_handler(triangulation)
      , mapping(fe_order)
      , x0(0.25, 0.25, 0.25)
      , model_sphere_center(0.0, 0.0, 0.0)
      , model_sphere_radius(1.0)
    {}


    Example2::Example2(const std::string &mesh_file_name, unsigned int fe_order)
      : mesh_file_name(mesh_file_name)
      , fe_order(fe_order)
      , fe(fe_order)
      , dof_handler(triangulation)
      , mapping(fe_order)
      , x0(0.25, 0.25, 0.25)
      , model_sphere_center(0.0, 0.0, 0.0)
      , model_sphere_radius(1.0)
    {}


    Example2::~Example2()
    {
      dof_handler.clear();
    }

    void
    Example2::generate_mesh(unsigned int number_of_refinements)
    {
      // Generate the initial mesh.
      GridGenerator::hyper_sphere(triangulation,
                                  model_sphere_center,
                                  model_sphere_radius);

      // Output the initial mesh.
      GridOut       grid_out;
      std::string   base_name("sphere-");
      std::ofstream mesh_file(base_name + std::string("0.msh"));
      grid_out.write_msh(triangulation, mesh_file);

      // Refine the mesh.
      for (unsigned int i = 0; i < number_of_refinements; i++)
        {
          triangulation.refine_global(1);
          std::ofstream mesh_file(base_name + std::to_string(i + 1) +
                                  std::string(".msh"));
          grid_out.write_msh(triangulation, mesh_file);
        }
    }


    void
    Example2::read_mesh()
    {
      GridIn<2, 3> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::fstream mesh_file(mesh_file_name);
      grid_in.read_msh(mesh_file);
    }


    void
    Example2::calc_cell_neighboring_types()
    {
      const unsigned int       n_active_cells = triangulation.n_active_cells();
      FullMatrix<unsigned int> cell_neighboring_type_matrix(n_active_cells,
                                                            n_active_cells);

      types::global_vertex_index i = 0;
      for (const auto &first_cell : triangulation.active_cell_iterators())
        {
          std::array<types::global_vertex_index,
                     GeometryInfo<2>::vertices_per_cell>
            first_cell_vertex_indices(get_vertex_indices<2, 3>(first_cell));

          types::global_vertex_index j = 0;
          for (const auto &second_cell : triangulation.active_cell_iterators())
            {
              std::array<types::global_vertex_index,
                         GeometryInfo<2>::vertices_per_cell>
                second_cell_vertex_indices(
                  get_vertex_indices<2, 3>(second_cell));

              std::vector<types::global_vertex_index> vertex_index_intersection;
              vertex_index_intersection.reserve(
                GeometryInfo<2>::vertices_per_cell);
              cell_neighboring_type_matrix(i, j) =
                LaplaceBEM::detect_cell_neighboring_type<2>(
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
      dof_handler.distribute_dofs(fe);

      const unsigned int n_dofs = dof_handler.n_dofs();

      system_matrix.reinit(n_dofs, n_dofs);
      system_rhs_matrix.reinit(n_dofs, n_dofs);
      system_rhs.reinit(n_dofs);
      neumann_bc.reinit(n_dofs);
      analytical_solution.reinit(n_dofs);
      solution.reinit(n_dofs);

      // Interpolate analytical solution values.
      AnalyticalSolution analytical_solution_function(x0);
      VectorTools::interpolate(dof_handler,
                               analytical_solution_function,
                               analytical_solution);

      // Interpolate Neumann BC values.
      NeumannBC neumann_bc_function(x0,
                                    model_sphere_center,
                                    model_sphere_radius);
      VectorTools::interpolate(dof_handler, neumann_bc_function, neumann_bc);
    }


    void
    Example2::assemble_on_one_cell(
      const typename DoFHandler<2, 3>::active_cell_iterator &cell_iter,
      CellWiseScratchData &                                  scratch,
      CellWisePerTaskData &                                  data)
    {
      // Clear the local matrix in case that it is reused from another finished
      // task. N.B. Its memory has already been allocated in the constructor of
      // <code>CellWisePerTaskData</code>.
      data.local_matrix = 0.;
      // N.B. The construction of the object <code>scratch.fe_values</code> is
      // carried out in the constructor of <code>CellWiseScratchData</code>
      scratch.fe_values.reinit(cell_iter);

      const unsigned int n_q_points = scratch.fe_values.get_quadrature().size();
      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      for (unsigned int q = 0; q < n_q_points; q++)
        {
          // Iterate over test function DoFs.
          for (unsigned int i = 0; i < dofs_per_cell; i++)
            {
              // Iterate over ansatz function DoFs.
              for (unsigned int j = 0; j < dofs_per_cell; j++)
                {
                  data.local_matrix(i, j) +=
                    0.5 * scratch.fe_values.shape_value(i, q) *
                    scratch.fe_values.shape_value(j, q) *
                    scratch.fe_values.JxW(q);
                }
            }
        }

      // Extract the DoF indices. N.B. Before calling
      // <code>get_dof_indices</code>, the memory for the argument vector should
      // have been allocated. Here, the memory for
      // <code>data.local_dof_indices</code> has been allocated in the
      // constructor of <code>CellWisePerTaskData</code>.
      cell_iter->get_dof_indices(data.local_dof_indices);
    }


    void
    Example2::copy_cell_local_to_global(const CellWisePerTaskData &data)
    {
      const unsigned int dofs_per_cell = data.local_matrix.m();

      // Assemble the local matrix to system matrix.
      for (unsigned int i = 0; i < dofs_per_cell; i++)
        {
          for (unsigned int j = 0; j < dofs_per_cell; j++)
            {
              system_matrix.add(data.local_dof_indices[i],
                                data.local_dof_indices[j],
                                data.local_matrix(i, j));
            }
        }
    }


    void
    Example2::assemble_on_one_pair_of_cells(
      const typename DoFHandler<2, 3>::active_cell_iterator &kx_cell_iter,
      const typename DoFHandler<2, 3>::active_cell_iterator &ky_cell_iter,
      const BEMValues<2, 3> &                                bem_values,
      PairCellWiseScratchData &                              scratch,
      PairCellWisePerTaskData &                              data)
    {
      // Geometry information.
      const unsigned int vertices_per_cell = GeometryInfo<2>::vertices_per_cell;

      // Determine the cell neighboring type based on the vertex dof indices.
      // The common dof indices will be stored into the vector
      // <code>vertex_dof_index_intersection</code> if there is any.
      std::array<types::global_dof_index, vertices_per_cell>
        kx_vertex_dof_indices(get_vertex_dof_indices<2, 3>(kx_cell_iter));
      std::array<types::global_dof_index, vertices_per_cell>
        ky_vertex_dof_indices(get_vertex_dof_indices<2, 3>(ky_cell_iter));

      scratch.vertex_dof_index_intersection.clear();
      CellNeighboringType cell_neighboring_type =
        detect_cell_neighboring_type<2>(kx_vertex_dof_indices,
                                        ky_vertex_dof_indices,
                                        scratch.vertex_dof_index_intersection);

      const FiniteElement<2, 3> &kx_fe = kx_cell_iter->get_fe();
      const FiniteElement<2, 3> &ky_fe = ky_cell_iter->get_fe();

      const unsigned int kx_n_dofs = kx_fe.dofs_per_cell;
      const unsigned int ky_n_dofs = ky_fe.dofs_per_cell;

      // Support points of $K_x$ and $K_y$ in the default
      // hierarchical order.
      hierarchical_support_points_in_real_cell(
        kx_cell_iter,
        kx_fe,
        this->mapping,
        scratch.kx_support_points_hierarchical);
      hierarchical_support_points_in_real_cell(
        ky_cell_iter,
        ky_fe,
        this->mapping,
        scratch.ky_support_points_hierarchical);

      // N.B. The vector holding local DoF indices has to have the right size
      // before being passed to the function <code>get_dof_indices</code>.
      kx_cell_iter->get_dof_indices(scratch.kx_local_dof_indices_hierarchical);
      ky_cell_iter->get_dof_indices(scratch.ky_local_dof_indices_hierarchical);

      try
        {
          // Quadrature rule to be adopted depending on the cell neighboring
          // type.
          const QGauss<4> *active_quad_rule = nullptr;

          switch (cell_neighboring_type)
            {
              case SamePanel:
                {
                  Assert(scratch.vertex_dof_index_intersection.size() ==
                           vertices_per_cell,
                         ExcInternalError());

                  // Get support points in tensor product oder.
                  permute_vector(scratch.kx_support_points_hierarchical,
                                 scratch.kx_fe_poly_space_numbering_inverse,
                                 scratch.kx_support_points_permuted);
                  permute_vector(scratch.ky_support_points_hierarchical,
                                 scratch.ky_fe_poly_space_numbering_inverse,
                                 scratch.ky_support_points_permuted);

                  // Get permuted local DoF indices.
                  permute_vector(scratch.kx_local_dof_indices_hierarchical,
                                 scratch.kx_fe_poly_space_numbering_inverse,
                                 data.kx_local_dof_indices_permuted);
                  permute_vector(scratch.ky_local_dof_indices_hierarchical,
                                 scratch.ky_fe_poly_space_numbering_inverse,
                                 data.ky_local_dof_indices_permuted);

                  active_quad_rule = &(bem_values.quad_rule_for_same_panel);

                  // Precalculate surface Jacobians and normal vectors.
                  for (unsigned int k3_index = 0; k3_index < 8; k3_index++)
                    {
                      for (unsigned int q = 0; q < active_quad_rule->size();
                           q++)
                        {
                          scratch.kx_jacobians_same_panel(k3_index, q) =
                            surface_jacobian_det_and_normal_vector(
                              k3_index,
                              q,
                              bem_values
                                .kx_shape_grad_matrix_table_for_same_panel,
                              scratch.kx_support_points_permuted,
                              scratch.kx_normals_same_panel(k3_index, q));

                          scratch.ky_jacobians_same_panel(k3_index, q) =
                            surface_jacobian_det_and_normal_vector(
                              k3_index,
                              q,
                              bem_values
                                .ky_shape_grad_matrix_table_for_same_panel,
                              scratch.ky_support_points_permuted,
                              scratch.ky_normals_same_panel(k3_index, q));

                          scratch.kx_quad_points_same_panel(k3_index, q) =
                            transform_unit_to_permuted_real_cell(
                              k3_index,
                              q,
                              bem_values.kx_shape_value_table_for_same_panel,
                              scratch.kx_support_points_permuted);

                          scratch.ky_quad_points_same_panel(k3_index, q) =
                            transform_unit_to_permuted_real_cell(
                              k3_index,
                              q,
                              bem_values.ky_shape_value_table_for_same_panel,
                              scratch.ky_support_points_permuted);
                        }
                    }

                  break;
                }
              case CommonEdge:
                {
                  // This part handles the common edge case of Sauter's
                  // quadrature rule.
                  // 1. Get the DoF indices in tensor product order for $K_x$.
                  // 2. Get the DoF indices in reversed tensor product order for
                  // $K_x$.
                  // 3. Extract DoF indices only for cell vertices in $K_x$ and
                  // $K_y$. N.B. The DoF indices for the last two vertices are
                  // swapped, such that the four vertices are in clockwise or
                  // counter clockwise order.
                  // 4. Determine the starting vertex.

                  Assert(scratch.vertex_dof_index_intersection.size() ==
                           GeometryInfo<2>::vertices_per_face,
                         ExcInternalError());

                  permute_vector(scratch.kx_local_dof_indices_hierarchical,
                                 scratch.kx_fe_poly_space_numbering_inverse,
                                 data.kx_local_dof_indices_permuted);

                  permute_vector(
                    scratch.ky_local_dof_indices_hierarchical,
                    scratch.ky_fe_reversed_poly_space_numbering_inverse,
                    data.ky_local_dof_indices_permuted);

                  std::array<types::global_dof_index, vertices_per_cell>
                    kx_local_vertex_dof_indices_swapped;
                  get_vertex_dof_indices_swapped<2, 3>(
                    kx_fe,
                    data.kx_local_dof_indices_permuted,
                    kx_local_vertex_dof_indices_swapped);
                  std::array<types::global_dof_index, vertices_per_cell>
                    ky_local_vertex_dof_indices_swapped;
                  get_vertex_dof_indices_swapped<2, 3>(
                    ky_fe,
                    data.ky_local_dof_indices_permuted,
                    ky_local_vertex_dof_indices_swapped);

                  // Determine the starting vertex index in $K_x$ and $K_y$.
                  unsigned int kx_starting_vertex_index =
                    get_start_vertex_dof_index<vertices_per_cell>(
                      scratch.vertex_dof_index_intersection,
                      kx_local_vertex_dof_indices_swapped);
                  Assert(kx_starting_vertex_index < vertices_per_cell,
                         ExcInternalError());
                  unsigned int ky_starting_vertex_index =
                    get_start_vertex_dof_index<vertices_per_cell>(
                      scratch.vertex_dof_index_intersection,
                      ky_local_vertex_dof_indices_swapped);
                  Assert(ky_starting_vertex_index < vertices_per_cell,
                         ExcInternalError());

                  // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                  // starting from <code>kx_starting_vertex_index</code> or
                  // <code>ky_starting_vertex_index</code>.
                  generate_forward_dof_permutation(
                    kx_fe,
                    kx_starting_vertex_index,
                    scratch.kx_local_dof_permutation);
                  generate_backward_dof_permutation(
                    ky_fe,
                    ky_starting_vertex_index,
                    scratch.ky_local_dof_permutation);

                  permute_vector(scratch.kx_support_points_hierarchical,
                                 scratch.kx_local_dof_permutation,
                                 scratch.kx_support_points_permuted);
                  permute_vector(scratch.ky_support_points_hierarchical,
                                 scratch.ky_local_dof_permutation,
                                 scratch.ky_support_points_permuted);

                  permute_vector(scratch.kx_local_dof_indices_hierarchical,
                                 scratch.kx_local_dof_permutation,
                                 data.kx_local_dof_indices_permuted);
                  permute_vector(scratch.ky_local_dof_indices_hierarchical,
                                 scratch.ky_local_dof_permutation,
                                 data.ky_local_dof_indices_permuted);

                  active_quad_rule = &(bem_values.quad_rule_for_common_edge);

                  // Precalculate surface Jacobians and normal vectors.
                  for (unsigned int k3_index = 0; k3_index < 6; k3_index++)
                    {
                      for (unsigned int q = 0; q < active_quad_rule->size();
                           q++)
                        {
                          scratch.kx_jacobians_common_edge(k3_index, q) =
                            surface_jacobian_det_and_normal_vector(
                              k3_index,
                              q,
                              bem_values
                                .kx_shape_grad_matrix_table_for_common_edge,
                              scratch.kx_support_points_permuted,
                              scratch.kx_normals_common_edge(k3_index, q));

                          scratch.ky_jacobians_common_edge(k3_index, q) =
                            surface_jacobian_det_and_normal_vector(
                              k3_index,
                              q,
                              bem_values
                                .ky_shape_grad_matrix_table_for_common_edge,
                              scratch.ky_support_points_permuted,
                              scratch.ky_normals_common_edge(k3_index, q));

                          scratch.kx_quad_points_common_edge(k3_index, q) =
                            transform_unit_to_permuted_real_cell(
                              k3_index,
                              q,
                              bem_values.kx_shape_value_table_for_common_edge,
                              scratch.kx_support_points_permuted);

                          scratch.ky_quad_points_common_edge(k3_index, q) =
                            transform_unit_to_permuted_real_cell(
                              k3_index,
                              q,
                              bem_values.ky_shape_value_table_for_common_edge,
                              scratch.ky_support_points_permuted);
                        }
                    }

                  break;
                }
              case CommonVertex:
                {
                  Assert(scratch.vertex_dof_index_intersection.size() == 1,
                         ExcInternalError());

                  permute_vector(scratch.kx_local_dof_indices_hierarchical,
                                 scratch.kx_fe_poly_space_numbering_inverse,
                                 data.kx_local_dof_indices_permuted);

                  permute_vector(scratch.ky_local_dof_indices_hierarchical,
                                 scratch.ky_fe_poly_space_numbering_inverse,
                                 data.ky_local_dof_indices_permuted);

                  std::array<types::global_dof_index, vertices_per_cell>
                    kx_local_vertex_dof_indices_swapped;
                  get_vertex_dof_indices_swapped<2, 3>(
                    kx_fe,
                    data.kx_local_dof_indices_permuted,
                    kx_local_vertex_dof_indices_swapped);
                  std::array<types::global_dof_index, vertices_per_cell>
                    ky_local_vertex_dof_indices_swapped;
                  get_vertex_dof_indices_swapped<2, 3>(
                    ky_fe,
                    data.ky_local_dof_indices_permuted,
                    ky_local_vertex_dof_indices_swapped);

                  // Determine the starting vertex index in $K_x$ and $K_y$.
                  unsigned int kx_starting_vertex_index =
                    get_start_vertex_dof_index<vertices_per_cell>(
                      scratch.vertex_dof_index_intersection,
                      kx_local_vertex_dof_indices_swapped);
                  Assert(kx_starting_vertex_index < vertices_per_cell,
                         ExcInternalError());
                  unsigned int ky_starting_vertex_index =
                    get_start_vertex_dof_index<vertices_per_cell>(
                      scratch.vertex_dof_index_intersection,
                      ky_local_vertex_dof_indices_swapped);
                  Assert(ky_starting_vertex_index < vertices_per_cell,
                         ExcInternalError());

                  // Generate the permutation of DoFs in $K_x$ and $K_y$ by
                  // starting from <code>kx_starting_vertex_index</code> or
                  // <code>ky_starting_vertex_index</code>.
                  generate_forward_dof_permutation(
                    kx_fe,
                    kx_starting_vertex_index,
                    scratch.kx_local_dof_permutation);
                  generate_forward_dof_permutation(
                    ky_fe,
                    ky_starting_vertex_index,
                    scratch.ky_local_dof_permutation);

                  permute_vector(scratch.kx_support_points_hierarchical,
                                 scratch.kx_local_dof_permutation,
                                 scratch.kx_support_points_permuted);
                  permute_vector(scratch.ky_support_points_hierarchical,
                                 scratch.ky_local_dof_permutation,
                                 scratch.ky_support_points_permuted);

                  permute_vector(scratch.kx_local_dof_indices_hierarchical,
                                 scratch.kx_local_dof_permutation,
                                 data.kx_local_dof_indices_permuted);
                  permute_vector(scratch.ky_local_dof_indices_hierarchical,
                                 scratch.ky_local_dof_permutation,
                                 data.ky_local_dof_indices_permuted);

                  active_quad_rule = &(bem_values.quad_rule_for_common_vertex);

                  // Precalculate surface Jacobians and normal vectors.
                  for (unsigned int k3_index = 0; k3_index < 4; k3_index++)
                    {
                      for (unsigned int q = 0; q < active_quad_rule->size();
                           q++)
                        {
                          scratch.kx_jacobians_common_vertex(k3_index, q) =
                            surface_jacobian_det_and_normal_vector(
                              k3_index,
                              q,
                              bem_values
                                .kx_shape_grad_matrix_table_for_common_vertex,
                              scratch.kx_support_points_permuted,
                              scratch.kx_normals_common_vertex(k3_index, q));

                          scratch.ky_jacobians_common_vertex(k3_index, q) =
                            surface_jacobian_det_and_normal_vector(
                              k3_index,
                              q,
                              bem_values
                                .ky_shape_grad_matrix_table_for_common_vertex,
                              scratch.ky_support_points_permuted,
                              scratch.ky_normals_common_vertex(k3_index, q));

                          scratch.kx_quad_points_common_vertex(k3_index, q) =
                            transform_unit_to_permuted_real_cell(
                              k3_index,
                              q,
                              bem_values.kx_shape_value_table_for_common_vertex,
                              scratch.kx_support_points_permuted);

                          scratch.ky_quad_points_common_vertex(k3_index, q) =
                            transform_unit_to_permuted_real_cell(
                              k3_index,
                              q,
                              bem_values.ky_shape_value_table_for_common_vertex,
                              scratch.ky_support_points_permuted);
                        }
                    }

                  break;
                }
              case Regular:
                {
                  Assert(scratch.vertex_dof_index_intersection.size() == 0,
                         ExcInternalError());

                  permute_vector(scratch.kx_local_dof_indices_hierarchical,
                                 scratch.kx_fe_poly_space_numbering_inverse,
                                 data.kx_local_dof_indices_permuted);

                  permute_vector(scratch.ky_local_dof_indices_hierarchical,
                                 scratch.ky_fe_poly_space_numbering_inverse,
                                 data.ky_local_dof_indices_permuted);

                  permute_vector(scratch.kx_support_points_hierarchical,
                                 scratch.kx_fe_poly_space_numbering_inverse,
                                 scratch.kx_support_points_permuted);
                  permute_vector(scratch.ky_support_points_hierarchical,
                                 scratch.ky_fe_poly_space_numbering_inverse,
                                 scratch.ky_support_points_permuted);

                  active_quad_rule = &(bem_values.quad_rule_for_regular);

                  // Precalculate surface Jacobians and normal vectors.
                  for (unsigned int q = 0; q < active_quad_rule->size(); q++)
                    {
                      scratch.kx_jacobians_regular(0, q) =
                        surface_jacobian_det_and_normal_vector(
                          0,
                          q,
                          bem_values.kx_shape_grad_matrix_table_for_regular,
                          scratch.kx_support_points_permuted,
                          scratch.kx_normals_regular(0, q));

                      scratch.ky_jacobians_regular(0, q) =
                        surface_jacobian_det_and_normal_vector(
                          0,
                          q,
                          bem_values.ky_shape_grad_matrix_table_for_regular,
                          scratch.ky_support_points_permuted,
                          scratch.ky_normals_regular(0, q));

                      scratch.kx_quad_points_regular(0, q) =
                        transform_unit_to_permuted_real_cell(
                          0,
                          q,
                          bem_values.kx_shape_value_table_for_regular,
                          scratch.kx_support_points_permuted);

                      scratch.ky_quad_points_regular(0, q) =
                        transform_unit_to_permuted_real_cell(
                          0,
                          q,
                          bem_values.ky_shape_value_table_for_regular,
                          scratch.ky_support_points_permuted);
                    }

                  break;
                }
              default:
                {
                  Assert(false, ExcNotImplemented());
                  active_quad_rule = nullptr;
                }
            }

          // Clear the local matrix in case that it is reused from another
          // finished task. N.B. Its memory has already been allocated in the
          // constructor of <code>CellPairWisePerTaskData</code>.
          data.dlp_matrix = 0.;
          data.slp_matrix = 0.;

          // Iterate over DoFs for test function space in tensor product
          // order in $K_x$.
          for (unsigned int i = 0; i < kx_n_dofs; i++)
            {
              // Iterate over DoFs for ansatz function space in tensor
              // product order in $K_y$.
              for (unsigned int j = 0; j < ky_n_dofs; j++)
                {
                  // Pullback the kernel function to unit cell.
                  KernelPulledbackToUnitCell<2, 3, double>
                    dlp_kernel_pullback_on_unit(
                      this->dlp,
                      cell_neighboring_type,
                      scratch.kx_support_points_permuted,
                      scratch.ky_support_points_permuted,
                      kx_fe,
                      ky_fe,
                      &bem_values,
                      &scratch,
                      i,
                      j);

                  // Pullback the kernel function to Sauter parameter
                  // space.
                  KernelPulledbackToSauterSpace<2, 3, double>
                    dlp_kernel_pullback_on_sauter(dlp_kernel_pullback_on_unit,
                                                  cell_neighboring_type,
                                                  &bem_values);

                  // Apply 4d Sauter numerical quadrature.
                  data.dlp_matrix(i, j) = ApplyQuadratureUsingBEMValues(
                    *active_quad_rule, dlp_kernel_pullback_on_sauter);

                  // Pullback the kernel function to unit cell.
                  KernelPulledbackToUnitCell<2, 3, double>
                    slp_kernel_pullback_on_unit(
                      this->slp,
                      cell_neighboring_type,
                      scratch.kx_support_points_permuted,
                      scratch.ky_support_points_permuted,
                      kx_fe,
                      ky_fe,
                      &bem_values,
                      &scratch,
                      i,
                      j);

                  // Pullback the kernel function to Sauter parameter
                  // space.
                  KernelPulledbackToSauterSpace<2, 3, double>
                    slp_kernel_pullback_on_sauter(slp_kernel_pullback_on_unit,
                                                  cell_neighboring_type,
                                                  &bem_values);

                  // Apply 4d Sauter numerical quadrature.
                  data.slp_matrix(i, j) = ApplyQuadratureUsingBEMValues(
                    *active_quad_rule, slp_kernel_pullback_on_sauter);
                }
            }
        }
      catch (const std::bad_cast &e)
        {
          Assert(false, ExcInternalError());
        }
    }

    void
    Example2::copy_pair_of_cells_local_to_global(
      const PairCellWisePerTaskData &data)
    {
      const unsigned int kx_dofs_per_cell = data.dlp_matrix.m();
      const unsigned int ky_dofs_per_cell = data.dlp_matrix.n();

      // Assemble the local matrix to system matrix.
      for (unsigned int i = 0; i < kx_dofs_per_cell; i++)
        {
          for (unsigned int j = 0; j < ky_dofs_per_cell; j++)
            {
              system_matrix.add(data.kx_local_dof_indices_permuted[i],
                                data.ky_local_dof_indices_permuted[j],
                                data.dlp_matrix(i, j));

              system_rhs_matrix.add(data.kx_local_dof_indices_permuted[i],
                                    data.ky_local_dof_indices_permuted[j],
                                    data.slp_matrix(i, j));
            }
        }
    }

    void
    Example2::assemble_system_smp()
    {
      MultithreadInfo::set_thread_limit(4);

      // Generate normal Gauss-Legendre quadrature rule for FEM integration.
      QGauss<2> quadrature_formula_2d(fe.degree + 1);

#ifdef GRAPH_COLORING
      // Graph coloring of the mesh for parallelizing matrix assembly.
      std::vector<std::vector<typename DoFHandler<2, 3>::active_cell_iterator>>
        colored_cells = GraphColoring::make_graph_coloring(
          dof_handler.begin_active(),
          dof_handler.end(),
          (std::function<std::vector<types::global_dof_index>(
             const typename DoFHandler<2, 3>::active_cell_iterator &)>)
            get_conflict_indices<2, 3>);

      WorkStream::run(colored_cells,
                      std::bind(&Example2::assemble_on_one_cell,
                                this,
                                std::placeholders::_1,
                                std::placeholders::_2,
                                std::placeholders::_3),
                      std::bind(&Example2::copy_cell_local_to_global,
                                this,
                                std::placeholders::_1),
                      CellWiseScratchData(fe,
                                          quadrature_formula_2d,
                                          update_values | update_JxW_values),
                      CellWisePerTaskData(fe));
#else
      WorkStream::run(dof_handler.begin_active(),
                      dof_handler.end(),
                      std::bind(&Example2::assemble_on_one_cell,
                                this,
                                std::placeholders::_1,
                                std::placeholders::_2,
                                std::placeholders::_3),
                      std::bind(&Example2::copy_cell_local_to_global,
                                this,
                                std::placeholders::_1),
                      CellWiseScratchData(fe,
                                          quadrature_formula_2d,
                                          update_values | update_JxW_values),
                      CellWisePerTaskData(fe));
#endif



      // Precalculate shape function values and their gradient
      // values at each quadrature point. N.B.
      // 1. The data tables for shape function values and their gradient values
      // should be calculated for both function space on $K_x$ and function
      // space on $K_y$.
      // 2. Being different from the integral in FEM, the integral in BEM
      // handled by Sauter's quadrature rule has multiple parts of $k_3$ (except
      // the regular cell neighboring type), each of which should be evaluated
      // at a different set of quadrature points in the unit cell after
      // coordinate transformation from the parametric space. Therefore, a
      // dimension with respect to $k_3$ term index should be added to the data
      // table compared to the usual FEValues.

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
      BEMValues<2, 3> bem_values(fe,
                                 fe,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      // Initialize the progress display.
      boost::progress_display pd(triangulation.n_active_cells(),
                                 deallog.get_console());

      PairCellWiseScratchData scratch_data(fe, fe, bem_values);
      PairCellWisePerTaskData per_task_data(fe, fe);
      // Calculate the term $(v, Ku)$ to be assembled into system matrix and
      // $(v, V(\psi))$ to be assembled into right-hand side vector.
      for (const auto &e : dof_handler.active_cell_iterators())
        {
#ifdef GRAPH_COLORING
          WorkStream::run(
            colored_cells,
            std::bind(&Example2::assemble_on_one_pair_of_cells,
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
            dof_handler.begin_active(),
            dof_handler.end(),
            std::bind(&Example2::assemble_on_one_pair_of_cells,
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
#endif

          ++pd;
        }

      // Calculate the right-hand side vector.
      system_rhs_matrix.vmult(system_rhs, neumann_bc);
    }


    void
    Example2::assemble_system_serial()
    {
      // Generate normal Gauss-Legendre quadrature rule for FEM integration.
      QGauss<2> quadrature_formula_2d(fe.degree + 1);

      const unsigned int dofs_per_cell = fe.dofs_per_cell;

      FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

      // Calculate the term $(v, \frac{1}{2}u)$.
      FEValues<2, 3> fe_values(fe,
                               quadrature_formula_2d,
                               update_values | update_JxW_values);

      // The memory of this vector should be allocated before the calling of
      // <code>get_dof_indices</code>.
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      const unsigned int n_q_points = quadrature_formula_2d.size();

      // Loop over each active cell.
      for (const auto &cell : dof_handler.active_cell_iterators())
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
                  system_matrix.add(local_dof_indices[i],
                                    local_dof_indices[j],
                                    local_matrix(i, j));
                }
            }
        }

      // Precalculate shape function values and their gradient
      // values at each quadrature point. N.B.
      // 1. The data tables for shape function values and their gradient values
      // should be calculated for both function space on $K_x$ and function
      // space on $K_y$.
      // 2. Being different from the integral in FEM, the integral in BEM
      // handled by Sauter's quadrature rule has multiple parts of $k_3$ (except
      // the regular cell neighboring type), each of which should be evaluated
      // at a different set of quadrature points in the unit cell after
      // coordinate transformation from the parametric space. Therefore, a
      // dimension with respect to $k_3$ term index should be added to the data
      // table compared to the usual FEValues.

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
      BEMValues<2, 3> bem_values(fe,
                                 fe,
                                 quad_rule_for_same_panel,
                                 quad_rule_for_common_edge,
                                 quad_rule_for_common_vertex,
                                 quad_rule_for_regular);
      bem_values.fill_shape_value_tables();
      bem_values.fill_shape_grad_matrix_tables();

      // Initialize the progress display.
      boost::progress_display pd(triangulation.n_active_cells() *
                                   triangulation.n_active_cells(),
                                 deallog.get_console());

      // Calculate the term $(v, Ku)$ to be assembled into system matrix and
      // $(v, V(\psi))$ to be assembled into right-hand side vector.
      for (const auto &e : dof_handler.active_cell_iterators())
        {
          for (const auto &f : dof_handler.active_cell_iterators())
            {
              SauterQuadRule(system_matrix,
                             this->dlp,
                             bem_values,
                             e,
                             f,
                             this->mapping,
                             this->mapping);

              SauterQuadRule(system_rhs_matrix,
                             this->slp,
                             bem_values,
                             e,
                             f,
                             this->mapping,
                             this->mapping);

              ++pd;
            }
        }

      // Calculate the right-hand side vector.
      system_rhs_matrix.vmult(system_rhs, neumann_bc);
    }

    void
    Example2::solve()
    {
      // Solve the system matrix using CG.
      SolverControl solver_control(1000, 1e-12);
      SolverCG<>    solver(solver_control);

      solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    }

    void
    Example2::output_results()
    {
      deallog << "Analytical solution:" << std::endl;
      analytical_solution.print(deallog.get_console(), 5);
      deallog << "Numerical solution:" << std::endl;
      solution.print(deallog.get_console(), 5);

      DataOut<2, DoFHandler<2, 3>> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(analytical_solution, "analytical_solution");
      data_out.add_data_vector(solution, "numerical_solution");
      data_out.build_patches();

      std::ofstream output("solution.vtk");
      data_out.write_vtk(output);
    }

    void
    Example2::run()
    {
      // generate_mesh(1);
      read_mesh();
      calc_cell_neighboring_types();
      setup_system();
#ifdef RUN_SMP_PARALLEL
      assemble_system_smp();
#else
      assemble_system_serial();
#endif
      solve();
      output_results();
    }
  } // namespace Erichsen1996Efficient
} // namespace LaplaceBEM

#endif /* INCLUDE_ERICHSEN1996EFFICIENT_EXAMPLE2_H_ */
