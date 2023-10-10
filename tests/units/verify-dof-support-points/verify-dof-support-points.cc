/**
 * \file verify-dof-support-points.cc
 * \brief Verify DoF support points for typical finite elements.
 *
 * \ingroup testers dealii_verify
 * \author Jihuan Tian
 * \date 2022-05-27
 */

#include <deal.II/base/logstream.h>
// H1-conforming finite element shape functions.
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iostream>

using namespace dealii;

namespace dealii
{
  template <int dim, int spacedim>
  void
  print_support_point_info(const MappingQ<dim, spacedim> &  mapping,
                           const DoFHandler<dim, spacedim> &dof_handler,
                           const std::string &              fe_name)
  {
    if (dof_handler.get_fe().has_support_points())
      {
        // Allocate memory for the vector storing support points.
        std::map<types::global_dof_index, Point<spacedim>> support_points;
        DoFTools::map_dofs_to_support_points(mapping,
                                             dof_handler,
                                             support_points);
        std::ofstream gnuplot_file(fe_name + ".gpl");
        DoFTools::write_gnuplot_dof_support_point_info(gnuplot_file,
                                                       support_points);
      }
  }
} // namespace dealii

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  // Generate a 2D single cell.
  const unsigned int dim = 2;

  Triangulation<dim, dim> triangulation;

  GridGenerator::hyper_cube(triangulation);
  GridOut       grid_out;
  std::ofstream mesh_file("one-cell.msh");
  grid_out.write_msh(triangulation, mesh_file);

  // Generate the Q1 mapping class, which describes the mapping from reference
  // cell to real cell.
  const MappingQ<dim, dim> mapping(1);

  // Dof handler.
  DoFHandler<dim, dim> dof_handler;

  // FE_Q continuous Lagrange element.
  FE_Q<dim, dim> fe_q_element(3);
  dof_handler.clear();
  dof_handler.initialize(triangulation, fe_q_element);
  print_support_point_info(mapping, dof_handler, "fe_q");

  /**
   * FE_Q can only be used for polynomial degrees greater than zero. If you want
   * an element of polynomial degree zero, then it cannot be continuous and you
   * will want to use @p FE_DGQ<dim>(0).
   */
  //  FE_Q<dim, dim> fe_q_element0(0);
  //  dof_handler.clear();
  //  dof_handler.initialize(triangulation, fe_q_element0);
  //  print_support_point_info(mapping, dof_handler, "fe_q0");

  // FE_DGQ discontinuous Lagrange element.
  FE_DGQ<dim, dim> fe_dgq_element0(0);
  dof_handler.clear();
  dof_handler.initialize(triangulation, fe_dgq_element0);
  print_support_point_info(mapping, dof_handler, "fe_dgq0");

  FE_DGQ<dim, dim> fe_dgq_element(3);
  dof_handler.clear();
  dof_handler.initialize(triangulation, fe_dgq_element);
  print_support_point_info(mapping, dof_handler, "fe_dgq");

  // Elasticity finite element.
  FESystem<dim, dim> elasticity_element(FE_Q<dim>(1), dim);
  dof_handler.clear();
  dof_handler.initialize(triangulation, elasticity_element);
  print_support_point_info(mapping, dof_handler, "elasticity_element");

  // High order scalar elasticity finite element.
  FESystem<dim, dim> scalar_elasticity_element_order3(FE_Q<dim>(3), 1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, scalar_elasticity_element_order3);
  print_support_point_info(mapping,
                           dof_handler,
                           "scalar_elasticity_element_order3");

  // Mixed Laplacian finite element.
  FESystem<dim, dim> rt_element(FE_RaviartThomas<dim>(1), 1, FE_DGQ<dim>(1), 1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, rt_element);
  print_support_point_info(mapping, dof_handler, "rt_element");

  FESystem<dim, dim> ldg_equal_element(FESystem<dim>(FE_DGQ<dim>(1), dim),
                                       1,
                                       FE_DGQ<dim>(1),
                                       1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, ldg_equal_element);
  print_support_point_info(mapping, dof_handler, "ldg_equal_element");

  FESystem<dim, dim> ldg_unequal_element(FESystem<dim>(FE_DGQ<dim>(1), dim),
                                         1,
                                         FE_DGQ<dim>(2),
                                         1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, ldg_unequal_element);
  print_support_point_info(mapping, dof_handler, "ldg_unequal_element");

  FESystem<dim, dim> ldg_convoluted_element_1a(FE_DGQ<dim>(1), dim + 1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, ldg_convoluted_element_1a);
  print_support_point_info(mapping, dof_handler, "ldg_convoluted_element_1a");

  FESystem<dim, dim> ldg_convoluted_element_1b(FE_DGQ<dim>(1),
                                               dim,
                                               FE_DGQ<dim>(1),
                                               1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, ldg_convoluted_element_1b);
  print_support_point_info(mapping, dof_handler, "ldg_convoluted_element_1b");

  FESystem<dim, dim> ldg_convoluted_element_2(FE_DGQ<dim>(1),
                                              dim,
                                              FE_DGQ<dim>(2),
                                              1);
  dof_handler.clear();
  dof_handler.initialize(triangulation, ldg_convoluted_element_2);
  print_support_point_info(mapping, dof_handler, "ldg_convoluted_element_2");

  // Finite element for eddy current problem.
  FESystem<dim, dim> eddy_current_element(FESystem<dim>(FE_Q<dim>(1), dim),
                                          1, // A_r
                                          FESystem<dim>(FE_Q<dim>(1), dim),
                                          1, // A_i
                                          FE_Q<dim>(1),
                                          1, // phi_r
                                          FE_Q<dim>(1),
                                          1 // phi_i
  );
  dof_handler.clear();
  dof_handler.initialize(triangulation, eddy_current_element);
  print_support_point_info(mapping, dof_handler, "eddy_current_element");

  return 0;
}
