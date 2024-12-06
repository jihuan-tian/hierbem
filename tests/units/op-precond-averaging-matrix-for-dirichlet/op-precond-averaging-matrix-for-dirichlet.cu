/**
 * @file op-precond-averaging-matrix-for-dirichlet.cu
 * @brief Verify building the averaging matrix for operator preconditioning used
 * in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2024-12-05
 */

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>

#include "preconditioners/preconditioner_for_laplace_dirichlet.h"

using namespace HierBEM;
using namespace dealii;
using namespace std;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  // Define the primal space and dual space with respect to the single layer
  // potential operator.
  FE_DGQ<2, 3> fe_primal_space(0);
  FE_Q<2, 3>   fe_dual_space(1);

  // Generate the mesh.
  Triangulation<2, 3> tria(
    Triangulation<2, 3>::MeshSmoothing::limit_level_difference_at_vertices);
  GridGenerator::subdivided_hyper_cube(tria, 3, 0, 10);
  ofstream mesh_out("mesh.msh");
  GridOut  grid_out;
  grid_out.write_msh(tria, mesh_out);

  // Create the preconditioner.
  PreconditionerForLaplaceDirichlet<2, 3, double> precond(fe_primal_space,
                                                          fe_dual_space,
                                                          tria);

  // Build the averaging matrix.
  precond.build_averaging_matrix();

  // Export the refined mesh.
  ofstream refined_mesh_out("refined-mesh.msh");
  grid_out.write_msh(precond.get_triangulation(), refined_mesh_out);

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_averaging_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print matrix.
  precond.get_averaging_matrix().print_formatted(std::cout, 15, true, 25, "0");

  // Generate DoF support points on the primal mesh.
  MappingQ<2, 3>                              mapping(1);
  std::map<types::global_dof_index, Point<3>> support_points;
  DoFToolsExt::map_mg_dofs_to_support_points(
    mapping, precond.get_dof_handler_dual_space(), 0, support_points);
  ofstream support_out("support-points-primal.gpl");
  DoFTools::write_gnuplot_dof_support_point_info(support_out, support_points);
  support_out.close();

  // Generate DoF support points on the refined mesh.
  DoFToolsExt::map_mg_dofs_to_support_points(
    mapping, precond.get_dof_handler_dual_space(), 1, support_points);
  support_out.open("support-points-refined.gpl");
  DoFTools::write_gnuplot_dof_support_point_info(support_out, support_points);
  support_out.close();

  return 0;
}
