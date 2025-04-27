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
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>

#include "grid_out_ext.h"
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
  write_msh_correct(tria, mesh_out);
  mesh_out.close();

  // Refine the triangulation which is needed by the preconditioner.
  tria.refine_global();
  mesh_out.open("refined-mesh.msh");
  write_msh_correct(tria, mesh_out);
  mesh_out.close();

  // Create the preconditioner. Since we do not apply the preconditioner to the
  // system matrix in this case, the conversion between internal and external
  // DoF numberings is not needed. Therefore, we pass a dummy numbering to the
  // preconditioner's constructor.
  std::vector<types::global_dof_index>                    dummy_numbering;
  PreconditionerForLaplaceDirichlet<2, 3, double, double> precond(
    fe_primal_space, fe_dual_space, tria, dummy_numbering, dummy_numbering);

  // Build the mass matrix.
  precond.initialize_dof_handlers();
  precond.build_dof_to_cell_topology();
  precond.build_mass_matrix_on_refined_mesh(QGauss<2>(2));

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_mass_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the mass matrix.
  precond.get_mass_matrix().print_formatted(std::cout, 15, true, 25, "0");

  return 0;
}
