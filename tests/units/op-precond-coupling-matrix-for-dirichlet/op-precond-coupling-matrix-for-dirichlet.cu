/**
 * @file op-precond-coupling-matrix-for-dirichlet.cc
 * @brief Verify building the coupling matrix for operator preconditioning used
 * in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2024-12-03
 */

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

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
  GridGenerator::subdivided_hyper_cube(tria, 10, 0, 10);
  ofstream mesh_out("mesh.msh");
  GridOut  grid_out;
  grid_out.write_msh(tria, mesh_out);

  // Create the preconditioner.
  PreconditionerForLaplaceDirichlet<2, 3, double> precond(fe_primal_space,
                                                          fe_dual_space,
                                                          tria);

  // Build the coupling matrix.
  precond.build_coupling_matrix();

  // Export the refined mesh.
  ofstream refined_mesh_out("refined-mesh.msh");
  grid_out.write_msh(precond.get_triangulation(), refined_mesh_out);

  // Print the sparsity pattern.
  precond.get_coupling_matrix().print_formatted(std::cout, 15, true, 25, "0");

  return 0;
}
