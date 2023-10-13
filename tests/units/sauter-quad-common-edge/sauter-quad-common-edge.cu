/**
 * @file sauter-quad-common-edge.cu
 * @brief Verify and demonstrate Sauter quadrature performed on a pair of cells
 * for the common edge case.
 *
 * @date 2020-11-18
 * @author Jihuan Tian
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <laplace_bem.h>

#include <fstream>

#include "laplace_bem.h"
#include "sauter_quadrature.hcu"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  /**
   * Generate a mesh containing two cells with a common edge.
   */
  const unsigned int           dim      = 2;
  const unsigned int           spacedim = 3;
  Triangulation<dim, spacedim> triangulation;

  std::vector<unsigned int> rectangle_repetitions;
  rectangle_repetitions.push_back(2);
  rectangle_repetitions.push_back(1);
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            rectangle_repetitions,
                                            Point<dim>(0, 0),
                                            Point<dim>(2, 2));
  std::ofstream mesh_file("./double-cells.msh");
  GridOut       grid_out;
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Generate finite element, which is shared by both test and ansatz spaces.
   */
  const unsigned int  fe_order = 2;
  FE_Q<dim, spacedim> fe(fe_order);

  /**
   * Generate Dof handler.
   */
  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /**
   * Generate mapping objects and associated smart pointers to their internal
   * data. High order mapping is adopted just to make this demo non-trivial in
   * the mapping aspect.
   *
   * N.B. Two mapping objects should be defined for the pair of cells \f$K_x\f$
   * and \f$K_y\f$ respectively, because the two sets of quadrature points
   * defined in the unit cells \f$\hat{K}_x\f$ and \f$\hat{K}_y\f$ are
   * different.
   */
  const unsigned int                mapping_order = 2;
  MappingQGenericExt<dim, spacedim> mapping_test_space(mapping_order);
  MappingQGenericExt<dim, spacedim> mapping_ansatz_space(mapping_order);

  /**
   * Procedures of extracting pointers to the internal data of mapping:
   * 1.Get the pointer to the internal database in the parent class @p Mapping.
   * N.B. A dummy quadrature object is passed to the @p get_data function. The
   * @p UpdateFlags is set to @p update_default (it means no update),
   * which at the moment disables any memory allocation, because this
   * operation will be manually taken care of later on.
   * 2. Downcast the smart pointer of @p Mapping<dim, spacedim>::InternalDataBase to
   * @p MappingQGeneric<dim,spacedim>::InternalData by first unwrapping
   * the original smart pointer via @p static_cast then wrapping it again.
   */
  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    mapping_database_test_space =
      mapping_test_space.get_data(update_default, QGauss<dim>(1));
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    mapping_data_test_space =
      std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
        static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
          mapping_database_test_space.release()));

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    mapping_database_ansatz_space =
      mapping_ansatz_space.get_data(update_default, QGauss<dim>(1));
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    mapping_data_ansatz_space =
      std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
        static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
          mapping_database_ansatz_space.release()));

  /**
   * Create different Laplace kernel functions.
   */
  HierBEM::CUDAWrappers::LaplaceKernel::SingleLayerKernel<spacedim>        slp;
  HierBEM::CUDAWrappers::LaplaceKernel::DoubleLayerKernel<spacedim>        dlp;
  HierBEM::CUDAWrappers::LaplaceKernel::AdjointDoubleLayerKernel<spacedim> adlp;
  HierBEM::CUDAWrappers::LaplaceKernel::HyperSingularKernel<spacedim> hyper;

  /**
   * Generate 4D Gauss-Legendre quadrature rules for various cell neighboring
   * types. Even though only the common edge case is considered in this
   * testcase, all of these quadrature objects are needed to initialize the
   * @p BEMValues object.
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
   * Precalculate data tables for shape function values at quadrature points in
   * the reference cells.
   *
   * Here shape functions have two meanings:
   * 1. basis polynomials for spanning the finite element space on a cell;
   * 2. basis polynomials for approximating the mapping from the reference cell
   * to real cells.
   */
  HierBEM::BEMValues<dim, spacedim> bem_values(fe,
                                               fe,
                                               *mapping_data_test_space,
                                               *mapping_data_ansatz_space,
                                               quad_rule_for_same_panel,
                                               quad_rule_for_common_edge,
                                               quad_rule_for_common_vertex,
                                               quad_rule_for_regular);
  bem_values.shape_function_values_common_edge();

  /**
   * Create temporary scratch data and copy data.
   */
  PairCellWiseScratchData<dim, spacedim, double> scratch_data(
    fe, fe, mapping_test_space, mapping_ansatz_space, bem_values);
  PairCellWisePerTaskData<dim, spacedim, double> copy_data(fe, fe);

  DoFHandler<dim, spacedim>::active_cell_iterator cell_iter =
    dof_handler.begin_active();

  std::vector<DoFHandler<dim, spacedim>::active_cell_iterator> cell_iterators;
  for (auto iter : dof_handler.active_cell_iterators())
    {
      cell_iterators.push_back(iter);
    }

  /**
   * Compute the Sauter quadrature for each pair of cell-local shape functions.
   */
  LAPACKFullMatrixExt<double> slp_cell_matrix(fe.dofs_per_cell,
                                              fe.dofs_per_cell);

  for (unsigned int i = 0; i < slp_cell_matrix.m(); i++)
    {
      for (unsigned int j = 0; j < slp_cell_matrix.n(); j++)
        {
          slp_cell_matrix(i, j) =
            sauter_quadrature_on_one_pair_of_shape_functions(
              slp,
              1.0,
              i,
              j,
              cell_iterators[0],
              cell_iterators[1],
              mapping_test_space,
              mapping_ansatz_space,
              bem_values,
              scratch_data,
              copy_data);
        }
    }

  std::cout << "Cell matrix for single layer potential kernel:\n";
  slp_cell_matrix.print_formatted_to_mat(std::cout, "slp", 15, true, 25);

  LAPACKFullMatrixExt<double> dlp_cell_matrix(fe.dofs_per_cell,
                                              fe.dofs_per_cell);

  for (unsigned int i = 0; i < dlp_cell_matrix.m(); i++)
    {
      for (unsigned int j = 0; j < dlp_cell_matrix.n(); j++)
        {
          dlp_cell_matrix(i, j) =
            sauter_quadrature_on_one_pair_of_shape_functions(
              dlp,
              1.0,
              i,
              j,
              cell_iterators[0],
              cell_iterators[1],
              mapping_test_space,
              mapping_ansatz_space,
              bem_values,
              scratch_data,
              copy_data);
        }
    }

  std::cout << "Cell matrix for double layer potential kernel:\n";
  dlp_cell_matrix.print_formatted_to_mat(std::cout, "dlp", 15, true, 25);

  LAPACKFullMatrixExt<double> adlp_cell_matrix(fe.dofs_per_cell,
                                               fe.dofs_per_cell);

  for (unsigned int i = 0; i < adlp_cell_matrix.m(); i++)
    {
      for (unsigned int j = 0; j < adlp_cell_matrix.n(); j++)
        {
          adlp_cell_matrix(i, j) =
            sauter_quadrature_on_one_pair_of_shape_functions(
              adlp,
              1.0,
              i,
              j,
              cell_iterators[0],
              cell_iterators[1],
              mapping_test_space,
              mapping_ansatz_space,
              bem_values,
              scratch_data,
              copy_data);
        }
    }

  std::cout << "Cell matrix for adjoint double layer potential kernel:\n";
  adlp_cell_matrix.print_formatted_to_mat(std::cout, "adlp", 15, true, 25);

  LAPACKFullMatrixExt<double> hyper_cell_matrix(fe.dofs_per_cell,
                                                fe.dofs_per_cell);

  for (unsigned int i = 0; i < hyper_cell_matrix.m(); i++)
    {
      for (unsigned int j = 0; j < hyper_cell_matrix.n(); j++)
        {
          hyper_cell_matrix(i, j) =
            sauter_quadrature_on_one_pair_of_shape_functions(
              hyper,
              1.0,
              i,
              j,
              cell_iterators[0],
              cell_iterators[1],
              mapping_test_space,
              mapping_ansatz_space,
              bem_values,
              scratch_data,
              copy_data);
        }
    }

  std::cout << "Cell matrix for hyper-singular potential kernel:\n";
  hyper_cell_matrix.print_formatted_to_mat(std::cout, "hyper", 15, true, 25);

  dof_handler.clear();
}
