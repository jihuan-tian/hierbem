/**
 * @file op-precond-averaging-matrix-for-dirichlet.cu
 * @brief Verify building the averaging matrix for operator preconditioning used
 * in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2024-12-05
 */

#include <deal.II/base/point.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "debug_tools.h"
#include "preconditioners/preconditioner_for_laplace_dirichlet.h"

using namespace HierBEM;
using namespace dealii;
using namespace std;

void
print_weights_at_support_points_in_refined_mesh(
  const PreconditionerForLaplaceDirichlet<2, 3, double> &precond)
{
  std::cout
    << "# DoF index in dual mesh, DoF index in refined mesh, Support point coordinates x, y, z, Weight for dual basis"
    << std::endl;

  std::vector<types::global_dof_index> dof_indices_in_cell(
    precond.get_dof_handler_dual_space().get_fe().dofs_per_cell);
  const std::vector<Point<2>> &unit_support_points =
    precond.get_dof_handler_dual_space().get_fe().get_unit_support_points();

  MappingQ<2, 3> mapping(1);

  // Iterate over each cell in the primal mesh, which is equivalent to
  // iterating over each node in the dual mesh.
  types::global_dof_index dof_index_in_dual_mesh = 0;
  for (const auto &cell :
       precond.get_dof_handler_dual_space().mg_cell_iterators_on_level(0))
    {
      // Iterate over each child in the refined mesh of the current cell in the
      // primal mesh.
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(dof_indices_in_cell);

          // Iterate over each DoF index in the child cell.
          unsigned int d = 0;
          for (auto dof_index_in_refined_mesh : dof_indices_in_cell)
            {
              Point<3> support_point =
                mapping.transform_unit_to_real_cell(child,
                                                    unit_support_points[d]);

              std::cout << dof_index_in_dual_mesh << " "
                        << dof_index_in_refined_mesh << " " << support_point(0)
                        << " " << support_point(1) << " " << support_point(2)
                        << " "
                        << precond.get_averaging_matrix().el(
                             dof_index_in_dual_mesh, dof_index_in_refined_mesh)
                        << std::endl;

              d++;
            }
        }

      dof_index_in_dual_mesh++;
    }
}

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  // Define the primal space and dual space with respect to the single layer
  // potential operator.
  FE_DGQ<2, 3> fe_primal_space(0);
  FE_Q<2, 3>   fe_dual_space(1);

  // Generate the mesh. Because we are going to distribute DoFs on the two-level
  // multigrid required by the operator preconditioner, the triangulation object
  // should be constructed with a level difference limitation at vertices.
  Triangulation<2, 3> tria(
    Triangulation<2, 3>::MeshSmoothing::limit_level_difference_at_vertices);
  GridGenerator::subdivided_hyper_cube(tria, 3, 0, 10);
  ofstream mesh_out("mesh.msh");
  GridOut  grid_out;
  grid_out.write_msh(tria, mesh_out);

  // Create the preconditioner. Since we do not apply the preconditioner to the
  // system matrix in this case, the conversion between internal and external
  // DoF numberings is not needed. Therefore, we pass a dummy numbering to the
  // preconditioner's constructor.
  std::vector<types::global_dof_index>            dummy_numbering;
  PreconditionerForLaplaceDirichlet<2, 3, double> precond(
    fe_primal_space, fe_dual_space, tria, dummy_numbering, dummy_numbering);

  precond.get_triangulation().copy_triangulation(tria);
  precond.get_triangulation().refine_global();

  // Build the averaging matrix.
  precond.initialize_dof_handlers();
  precond.build_dof_to_cell_topology();
  precond.build_averaging_matrix();

  // Export the refined mesh.
  ofstream refined_mesh_out("refined-mesh.msh");
  grid_out.write_msh(precond.get_triangulation(), refined_mesh_out);

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_averaging_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the averaging matrix.
  print_sparse_matrix_to_mat(
    std::cout, "Cd", precond.get_averaging_matrix(), 15, true, 25);

  print_weights_at_support_points_in_refined_mesh(precond);

  return 0;
}
