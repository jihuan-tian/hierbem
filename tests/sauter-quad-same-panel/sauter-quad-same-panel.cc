#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <laplace_bem.h>

#include <fstream>


using namespace dealii;

#define WITH_BEM_VALUES

int
main()
{
  deallog.depth_console(2);
  deallog.pop();

  // Generate a single cell mesh.
  const unsigned int           dim      = 2;
  const unsigned int           spacedim = 3;
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_rectangle(triangulation,
                                 Point<dim>(0, 0),
                                 Point<dim>(1, 2));
  std::ofstream mesh_file("./single-cell.msh");
  GridOut       grid_out;
  grid_out.write_msh(triangulation, mesh_file);


  // Generate finite element.
  const unsigned int  fe_order = 2;
  FE_Q<dim, spacedim> fe(fe_order);
  // Generate Q1 mapping.
  const unsigned int             mapping_order = fe_order;
  MappingQGeneric<dim, spacedim> mapping(mapping_order);
  // Generate Dof handler.
  DoFHandler<dim, spacedim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Generate the single layer kernel function object.
  HierBEM::LaplaceKernel::SingleLayerKernel<spacedim> slp;
  // Generate the double layer kernel function object.
  HierBEM::LaplaceKernel::DoubleLayerKernel<spacedim> dlp;
  // Generate the adjoint double layer kernel function object.
  HierBEM::LaplaceKernel::AdjointDoubleLayerKernel<spacedim> adlp;
  // Generate the hyper-singular kernel function object.
  HierBEM::LaplaceKernel::HyperSingularKernel<spacedim> hyper;


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
  HierBEM::BEMValues<2, 3> bem_values(fe,
                                         fe,
                                         quad_rule_for_same_panel,
                                         quad_rule_for_common_edge,
                                         quad_rule_for_common_vertex,
                                         quad_rule_for_regular);
  bem_values.fill_shape_value_tables();
  bem_values.fill_shape_grad_matrix_tables();


  DoFHandler<dim, spacedim>::active_cell_iterator cell_iter =
    dof_handler.begin_active();

  FullMatrix<double> slp_cell_matrix;

#ifndef WITH_BEM_VALUES
  slp_cell_matrix =
    HierBEM::SauterQuadRule(slp, cell_iter, cell_iter, mapping, mapping);
#else
  slp_cell_matrix = HierBEM::SauterQuadRule(
    slp, bem_values, cell_iter, cell_iter, mapping, mapping);
#endif

  deallog << "Cell matrix for single layer potential kernel:\n";
  slp_cell_matrix.print(deallog, 12, 5);

  FullMatrix<double> dlp_cell_matrix;

#ifndef WITH_BEM_VALUES
  dlp_cell_matrix =
    HierBEM::SauterQuadRule(dlp, cell_iter, cell_iter, mapping, mapping);
#else
  dlp_cell_matrix = HierBEM::SauterQuadRule(
    dlp, bem_values, cell_iter, cell_iter, mapping, mapping);
#endif

  deallog << "Cell matrix for double layer potential kernel:\n";
  dlp_cell_matrix.print(deallog, 12, 5);

  FullMatrix<double> adlp_cell_matrix;

#ifndef WITH_BEM_VALUES
  adlp_cell_matrix =
    HierBEM::SauterQuadRule(adlp, cell_iter, cell_iter, mapping, mapping);
#else
  adlp_cell_matrix = HierBEM::SauterQuadRule(
    adlp, bem_values, cell_iter, cell_iter, mapping, mapping);
#endif

  deallog << "Cell matrix for adjoint double layer potential kernel:\n";
  adlp_cell_matrix.print(deallog, 12, 5);

  FullMatrix<double> hyper_cell_matrix;

#ifndef WITH_BEM_VALUES
  hyper_cell_matrix =
    HierBEM::SauterQuadRule(hyper, cell_iter, cell_iter, mapping, mapping);
#else
  hyper_cell_matrix = HierBEM::SauterQuadRule(
    hyper, bem_values, cell_iter, cell_iter, mapping, mapping);
#endif

  deallog << "Cell matrix for hyper-singular potential kernel:\n";
  hyper_cell_matrix.print(deallog, 12, 5);

  dof_handler.clear();
}
