/**
 * @file op-precond-mass-matrix-for-dirichlet.cu
 * @brief Verify building the mass matrix for operator preconditioning used
 * in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2024-12-06
 */

#include <deal.II/base/quadrature_lib.h>

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

  precond.get_triangulation().copy_triangulation(tria);
  precond.get_triangulation().refine_global();

  // Build the averaging matrix.
  precond.initialize_dof_handlers();
  precond.build_dof_to_cell_topology();
  precond.build_mass_matrix_on_refined_mesh(QGauss<2>(2));

  // Export the refined mesh.
  ofstream refined_mesh_out("refined-mesh.msh");
  grid_out.write_msh(precond.get_triangulation(), refined_mesh_out);

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_mass_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print matrix.
  precond.get_mass_matrix().print_formatted(std::cout, 15, true, 25, "0");

  return 0;
}
