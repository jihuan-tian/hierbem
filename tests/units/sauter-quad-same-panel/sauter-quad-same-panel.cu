/**
 * @file sauter-quad-same-panel.cu
 * @brief Verify and demonstrate Sauter quadrature performed on a pair of cells
 * for the same panel case.
 *
 * @date 2020-11-18
 * @author Jihuan Tian
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>

#include "laplace_bem.hcu"
#include "sauter_quadrature.hcu"

using namespace dealii;
using namespace HierBEM;

class OutwardSurfaceNormalDetector
{
public:
  // Because there is no need to invert the computed normal vector of a cell in
  // this test, this function always returns false.
  bool
  is_normal_vector_inward([[maybe_unused]] const types::material_id m) const
  {
    return false;
  }
};

// Initialize mapping objects from the first order to the maximum.
template <int dim, int spacedim>
void
initialize_mappings(std::vector<MappingInfo<dim, spacedim> *> &mappings,
                    const unsigned int max_mapping_order)
{
  // Create different orders of mapping.
  mappings.reserve(max_mapping_order);
  for (unsigned int i = 1; i <= max_mapping_order; i++)
    {
      mappings.push_back(new MappingInfo<dim, spacedim>(i));
    }
}

// Release all mappings on the heap.
template <int dim, int spacedim>
void
destroy_mappings(std::vector<MappingInfo<dim, spacedim> *> &mappings)
{
  for (auto m : mappings)
    {
      if (m != nullptr)
        delete m;
    }
}

int
main()
{
  /**
   * Generate a single cell mesh.
   */
  const unsigned int           dim      = 2;
  const unsigned int           spacedim = 3;
  Triangulation<dim, spacedim> triangulation;

  GridGenerator::hyper_rectangle(triangulation,
                                 Point<dim>(0, 0),
                                 Point<dim>(1, 2));

  std::ofstream mesh_file("./single-cell.msh");
  GridOut       grid_out;
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Generate mapping objects and associated smart pointers to their internal
   * data. High order mapping is adopted just to make this demo non-trivial in
   * the mapping aspect.
   *
   * N.B. Two mapping objects should be defined for the pair of cells
   * \f$K_x\f$ and \f$K_y\f$ respectively, because the two sets of quadrature
   * points defined in the unit cells \f$\hat{K}_x\f$ and \f$\hat{K}_y\f$ are
   * different.
   */
  const unsigned int                        max_mapping_order = 3;
  std::vector<MappingInfo<dim, spacedim> *> mappings;
  initialize_mappings(mappings, max_mapping_order);

  {
    std::cout << "=== fe-order=(dirichlet:2, neumann:2), mapping order=2 ==="
              << std::endl;
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

    const unsigned int          mapping_order = 2;
    MappingInfo<dim, spacedim> &mapping_info_test_space =
      *mappings[mapping_order - 1];
    MappingInfo<dim, spacedim> &mapping_info_ansatz_space =
      *mappings[mapping_order - 1];

    /**
     * Create different Laplace kernel functions.
     */
    HierBEM::CrossPlatform::LaplaceKernel::SingleLayerKernel<spacedim> slp;
    HierBEM::CrossPlatform::LaplaceKernel::DoubleLayerKernel<spacedim> dlp;
    HierBEM::CrossPlatform::LaplaceKernel::AdjointDoubleLayerKernel<spacedim>
                                                                        adlp;
    HierBEM::CrossPlatform::LaplaceKernel::HyperSingularKernel<spacedim> hyper;


    /**
     * Generate 4D Gauss-Legendre quadrature rules for various cell neighboring
     * types. Even though only the same panel case is considered in this
     * testcase,
     * all of these quadrature objects are needed to initialize the @p BEMValues
     * object.
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
     * Precalculate data tables for shape function values at quadrature points
     * in the reference cells.
     *
     * Here shape functions have two meanings:
     * 1. basis polynomials for spanning the finite element space on a cell;
     * 2. basis polynomials for approximating the mapping from the reference
     * cell to real cells.
     */
    HierBEM::BEMValues<dim, spacedim> bem_values(fe,
                                                 fe,
                                                 mappings,
                                                 quad_rule_for_same_panel,
                                                 quad_rule_for_common_edge,
                                                 quad_rule_for_common_vertex,
                                                 quad_rule_for_regular);
    bem_values.shape_function_values_same_panel();

    /**
     * Create temporary scratch data and copy data.
     */
    PairCellWiseScratchData<dim, spacedim, double> scratch_data(fe,
                                                                fe,
                                                                mappings,
                                                                bem_values);
    PairCellWisePerTaskData<dim, spacedim, double> copy_data(fe, fe);

    DoFHandler<dim, spacedim>::active_cell_iterator cell_iter =
      dof_handler.begin_active();

    /**
     * Compute the Sauter quadrature for each pair of cell-local shape
     * functions.
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
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                OutwardSurfaceNormalDetector(),
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
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                OutwardSurfaceNormalDetector(),
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
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                OutwardSurfaceNormalDetector(),
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
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                OutwardSurfaceNormalDetector(),
                scratch_data,
                copy_data);
          }
      }

    std::cout << "Cell matrix for hyper-singular potential kernel:\n";
    hyper_cell_matrix.print_formatted_to_mat(std::cout, "hyper", 15, true, 25);

    dof_handler.clear();

    scratch_data.release();
    copy_data.release();
  }

  {
    std::cout << "=== fe-order=(dirichlet:2, neumann:1), mapping order=2 ==="
              << std::endl;

    /**
     * Generate finite element, which is shared by both test and ansatz spaces.
     */
    FE_DGQ<dim, spacedim> fe_neumann_space(1);
    FE_Q<dim, spacedim>   fe_dirichlet_space(2);

    /**
     * Generate Dof handler.
     */
    DoFHandler<dim, spacedim> dof_handler_neumann_space(triangulation);
    DoFHandler<dim, spacedim> dof_handler_dirichlet_space(triangulation);
    dof_handler_neumann_space.distribute_dofs(fe_neumann_space);
    dof_handler_dirichlet_space.distribute_dofs(fe_dirichlet_space);

    const unsigned int          mapping_order = 2;
    MappingInfo<dim, spacedim> &mapping_info_test_space =
      *mappings[mapping_order - 1];
    MappingInfo<dim, spacedim> &mapping_info_ansatz_space =
      *mappings[mapping_order - 1];

    /**
     * Create different Laplace kernel functions.
     */
    HierBEM::CrossPlatform::LaplaceKernel::SingleLayerKernel<spacedim> slp;
    HierBEM::CrossPlatform::LaplaceKernel::DoubleLayerKernel<spacedim> dlp;
    HierBEM::CrossPlatform::LaplaceKernel::AdjointDoubleLayerKernel<spacedim>
                                                                        adlp;
    HierBEM::CrossPlatform::LaplaceKernel::HyperSingularKernel<spacedim> hyper;

    /**
     * Generate 4D Gauss-Legendre quadrature rules for various cell neighboring
     * types. Even though only the same panel case is considered in this
     * testcase,
     * all of these quadrature objects are needed to initialize the @p BEMValues
     * object.
     */
    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<4> quad_rule_for_regular(quad_order_for_regular);

    DoFHandler<dim, spacedim>::active_cell_iterator cell_iter_neumann_space =
      dof_handler_neumann_space.begin_active();
    DoFHandler<dim, spacedim>::active_cell_iterator cell_iter_dirichlet_space =
      dof_handler_dirichlet_space.begin_active();

    {
      HierBEM::BEMValues<dim, spacedim> bem_values(fe_neumann_space,
                                                   fe_neumann_space,
                                                   mappings,
                                                   quad_rule_for_same_panel,
                                                   quad_rule_for_common_edge,
                                                   quad_rule_for_common_vertex,
                                                   quad_rule_for_regular);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchData<dim, spacedim, double> scratch_data(
        fe_neumann_space, fe_neumann_space, mappings, bem_values);
      PairCellWisePerTaskData<dim, spacedim, double> copy_data(
        fe_neumann_space, fe_neumann_space);

      /**
       * Compute the Sauter quadrature for each pair of cell-local shape
       * functions.
       */
      LAPACKFullMatrixExt<double> slp_cell_matrix(
        fe_neumann_space.dofs_per_cell, fe_neumann_space.dofs_per_cell);

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
                  cell_iter_neumann_space,
                  cell_iter_neumann_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  OutwardSurfaceNormalDetector(),
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for single layer potential kernel:\n";
      slp_cell_matrix.print_formatted_to_mat(std::cout, "slp", 15, true, 25);

      scratch_data.release();
      copy_data.release();
    }

    {
      HierBEM::BEMValues<dim, spacedim> bem_values(fe_neumann_space,
                                                   fe_dirichlet_space,
                                                   mappings,
                                                   quad_rule_for_same_panel,
                                                   quad_rule_for_common_edge,
                                                   quad_rule_for_common_vertex,
                                                   quad_rule_for_regular);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchData<dim, spacedim, double> scratch_data(
        fe_neumann_space, fe_dirichlet_space, mappings, bem_values);
      PairCellWisePerTaskData<dim, spacedim, double> copy_data(
        fe_neumann_space, fe_dirichlet_space);

      LAPACKFullMatrixExt<double> dlp_cell_matrix(
        fe_neumann_space.dofs_per_cell, fe_dirichlet_space.dofs_per_cell);

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
                  cell_iter_neumann_space,
                  cell_iter_dirichlet_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  OutwardSurfaceNormalDetector(),
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for double layer potential kernel:\n";
      dlp_cell_matrix.print_formatted_to_mat(std::cout, "dlp", 15, true, 25);

      scratch_data.release();
      copy_data.release();
    }

    {
      HierBEM::BEMValues<dim, spacedim> bem_values(fe_dirichlet_space,
                                                   fe_neumann_space,
                                                   mappings,
                                                   quad_rule_for_same_panel,
                                                   quad_rule_for_common_edge,
                                                   quad_rule_for_common_vertex,
                                                   quad_rule_for_regular);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchData<dim, spacedim, double> scratch_data(
        fe_dirichlet_space, fe_neumann_space, mappings, bem_values);
      PairCellWisePerTaskData<dim, spacedim, double> copy_data(
        fe_dirichlet_space, fe_neumann_space);

      LAPACKFullMatrixExt<double> adlp_cell_matrix(
        fe_dirichlet_space.dofs_per_cell, fe_neumann_space.dofs_per_cell);

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
                  cell_iter_dirichlet_space,
                  cell_iter_neumann_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  OutwardSurfaceNormalDetector(),
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for adjoint double layer potential kernel:\n";
      adlp_cell_matrix.print_formatted_to_mat(std::cout, "adlp", 15, true, 25);

      scratch_data.release();
      copy_data.release();
    }

    {
      HierBEM::BEMValues<dim, spacedim> bem_values(fe_dirichlet_space,
                                                   fe_dirichlet_space,
                                                   mappings,
                                                   quad_rule_for_same_panel,
                                                   quad_rule_for_common_edge,
                                                   quad_rule_for_common_vertex,
                                                   quad_rule_for_regular);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchData<dim, spacedim, double> scratch_data(
        fe_dirichlet_space, fe_dirichlet_space, mappings, bem_values);
      PairCellWisePerTaskData<dim, spacedim, double> copy_data(
        fe_dirichlet_space, fe_dirichlet_space);

      LAPACKFullMatrixExt<double> hyper_cell_matrix(
        fe_dirichlet_space.dofs_per_cell, fe_dirichlet_space.dofs_per_cell);

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
                  cell_iter_dirichlet_space,
                  cell_iter_dirichlet_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  OutwardSurfaceNormalDetector(),
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for hyper-singular potential kernel:\n";
      hyper_cell_matrix.print_formatted_to_mat(
        std::cout, "hyper", 15, true, 25);

      scratch_data.release();
      copy_data.release();
    }

    dof_handler_neumann_space.clear();
    dof_handler_dirichlet_space.clear();
  }

  destroy_mappings(mappings);
}
