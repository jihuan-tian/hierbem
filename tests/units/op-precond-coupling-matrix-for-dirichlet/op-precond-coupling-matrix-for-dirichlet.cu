/**
 * @file op-precond-coupling-matrix-for-dirichlet.cc
 * @brief Verify building the coupling matrix for operator preconditioning used
 * in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2024-12-03
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

#include "debug_tools.hcu"
#include "preconditioners/preconditioner_for_laplace_dirichlet.h"

using namespace HierBEM;
using namespace dealii;
using namespace std;

/**
 * @brief Print the DoF indices and support point coordinates in the given cell.
 *
 * Each line corresponds to one DoF and data fields are separated by white
 * space.
 * @param dof_handler
 * @param cell
 */
void
print_cell_support_point_info(const DoFHandler<2, 3> &dof_handler,
                              const DoFHandler<2, 3>::cell_iterator &cell)
{
  std::vector<types::global_dof_index> dof_indices_in_cell(
    dof_handler.get_fe().dofs_per_cell);
  cell->get_mg_dof_indices(dof_indices_in_cell);
  types::global_dof_index dof_index = dof_indices_in_cell[0];

  MappingQ<2, 3>               mapping(1);
  const std::vector<Point<2>> &unit_support_points =
    dof_handler.get_fe().get_unit_support_points();
  Point<2> unit_support_point = unit_support_points[0];
  Point<3> support_point =
    mapping.transform_unit_to_real_cell(cell, unit_support_point);

  std::cout << dof_index << " " << support_point(0) << " " << support_point(1)
            << " " << support_point(2) << std::endl;
}

/**
 * @brief Print DoF support points in each cell in the primal mesh, then follow
 * the support points in each of its child cell in the refined mesh.
 *
 * The double level multigrid and DoF handlers are stored in the given
 * preconditioner.
 * @param precond
 */
void
print_support_points_in_primal_and_refined_meshes(
  const PreconditionerForLaplaceDirichlet<2, 3, double> &precond)
{
  std::cout << "# DoF index, Support point coordinates x, y, z" << std::endl;

  const DoFHandler<2, 3> &dof_handler = precond.get_dof_handler_primal_space();
  // Iterate over each cell in the primal mesh.
  for (const auto &cell : dof_handler.mg_cell_iterators_on_level(0))
    {
      print_cell_support_point_info(dof_handler, cell);
      // Iterate over each child iterator of the current cell, i.e. on
      // level 1.
      for (const auto &child : cell->child_iterators())
        print_cell_support_point_info(dof_handler, child);
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

  // Build the coupling matrix.
  precond.initialize_dof_handlers();
  precond.build_dof_to_cell_topology();
  precond.build_coupling_matrix();

  // Export the refined mesh.
  ofstream refined_mesh_out("refined-mesh.msh");
  grid_out.write_msh(precond.get_triangulation(), refined_mesh_out);

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_coupling_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print matrix.
  print_sparse_matrix_to_mat(
    std::cout, "Cp", precond.get_coupling_matrix(), 15, true, 25);

  // Print DoF indices and support point coordinates for cells in the primal
  // mesh. Following each cell data, there are DoF indices and support point
  // coordinates for the four child cells in the refined mesh.
  print_support_points_in_primal_and_refined_meshes(precond);

  return 0;
}
