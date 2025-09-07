/**
 * @file op-precond-averaging-matrix-for-dirichlet-subdomain.cu
 * @brief Verify building the averaging matrix for operator preconditioning on a
 * subdomain used in Laplace Dirichlet problem.
 *
 * @ingroup preconditioners
 * @author Jihuan Tian
 * @date 2025-01-22
 */

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "grid/grid_out_ext.h"
#include "preconditioners/preconditioner_for_laplace_dirichlet.h"
#include "utilities/debug_tools.h"

using namespace Catch::Matchers;
using namespace HierBEM;
using namespace dealii;
using namespace std;

// For the 6x3 subdivided rectangle, we assign material id 1 to the left half
// 3x3 grid and 2 to the right half 3x3 grid. The former is assigned the
// Dirichlet boundary condition and the latter is assigned the Neumann boundary
// condition.
void
assign_material_ids(Triangulation<2, 3> &tria, const double width)
{
  for (auto &cell : tria.active_cell_iterators())
    {
      if (cell->center()(0) < width / 2.0)
        cell->set_material_id(1);
      else
        cell->set_material_id(2);
    }
}

void
setup_preconditioner(
  PreconditionerForLaplaceDirichlet<2, 3, double, double> &precond)
{
  precond.initialize_dof_handlers();
  precond.generate_dof_selectors();
  precond.generate_maps_between_full_and_local_dof_ids();
  precond.build_dof_to_cell_topology();
}

void
print_weights_at_support_points_in_refined_mesh(
  ostream                                                       &out,
  const PreconditionerForLaplaceDirichlet<2, 3, double, double> &precond)
{
  out
    << "# Local DoF index in dual mesh, Full DoF index in refined mesh, Local DoF index in refined mesh, Support point coordinates x, y, z, Weight for dual basis"
    << std::endl;

  std::vector<types::global_dof_index> dof_indices_in_cell(
    precond.get_dof_handler_dual_space().get_fe().dofs_per_cell);
  const std::vector<Point<2>> &unit_support_points =
    precond.get_dof_handler_dual_space().get_fe().get_unit_support_points();

  MappingQ<2, 3> mapping(1);

  // Iterate over each cell in the primal mesh in the subdomain, which is
  // equivalent to iterating over each node in the dual mesh.
  types::global_dof_index local_dof_index_in_dual_mesh = 0;
  for (const auto &cell :
       precond.get_dof_handler_dual_space().mg_cell_iterators_on_level(0))
    {
      auto found_iter =
        precond.get_subdomain_material_ids().find(cell->material_id());
      if (found_iter != precond.get_subdomain_material_ids().end())
        {
          // Iterate over each child in the refined mesh of the current cell in
          // the primal mesh.
          for (const auto &child : cell->child_iterators())
            {
              child->get_mg_dof_indices(dof_indices_in_cell);

              // Iterate over each DoF index in the child cell.
              unsigned int d = 0;
              for (auto dof_index_in_refined_mesh : dof_indices_in_cell)
                {
                  Assert(precond.get_dual_space_dof_selectors_on_refined_mesh()
                           [dof_index_in_refined_mesh],
                         ExcInternalError());

                  types::global_dof_index local_dof_index_in_refined_mesh =
                    precond
                      .get_dual_space_full_to_local_dof_id_map_on_refined_mesh()
                        [dof_index_in_refined_mesh];

                  Point<3> support_point =
                    mapping.transform_unit_to_real_cell(child,
                                                        unit_support_points[d]);

                  out << local_dof_index_in_dual_mesh << " "
                      << dof_index_in_refined_mesh << " "
                      << local_dof_index_in_refined_mesh << " "
                      << support_point(0) << " " << support_point(1) << " "
                      << support_point(2) << " "
                      << precond.get_averaging_matrix().el(
                           local_dof_index_in_dual_mesh,
                           local_dof_index_in_refined_mesh)
                      << std::endl;

                  d++;
                }
            }

          local_dof_index_in_dual_mesh++;
        }
    }
}

void
compare_two_files(const string &file1, const string &file2)
{
  vector<string> file1_lines, file2_lines;
  read_file_lines(file1, file1_lines);
  read_file_lines(file2, file2_lines);

  REQUIRE(file1_lines.size() == file2_lines.size());

  for (size_t i = 0; i < file1_lines.size(); i++)
    {
      INFO("File 1: " << file1 << "\n"
                      << "File 2: " << file2 << "\n"
                      << "Mismatch on line: " << i + 1);
      REQUIRE(file1_lines[i] == file2_lines[i]);
    }
}

TEST_CASE(
  "Verify averaging matrix for operator preconditioning in Laplace Dirichlet on subdomain",
  "[preconditioner]")
{
  INFO("*** test start");

  // Define the primal space and dual space with respect to the single layer
  // potential operator.
  FE_DGQ<2, 3> fe_primal_space(0);
  FE_Q<2, 3>   fe_dual_space(1);

  // Generate the mesh. Because we are going to distribute DoFs on the two-level
  // multigrid required by the operator preconditioner, the triangulation object
  // should be constructed with a level difference limitation at vertices.
  const double        width        = 12.0;
  const double        height       = 6.0;
  const unsigned int  x_repetition = 6;
  const unsigned int  y_repetition = 3;
  Triangulation<2, 3> tria(
    Triangulation<2, 3>::MeshSmoothing::limit_level_difference_at_vertices);
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {x_repetition, y_repetition},
                                            Point<2>(0, 0),
                                            Point<2>(width, height));
  assign_material_ids(tria, width);

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
  // preconditioner's constructor. Its size is initialized to be half of the
  // number of cells in the triangulation.
  std::vector<types::global_dof_index> dummy_numbering(tria.n_cells(0) / 2);
  std::set<types::material_id>         subdomain_material_ids = {1};
  PreconditionerForLaplaceDirichlet<2, 3, double, double> precond(
    fe_primal_space,
    fe_dual_space,
    tria,
    dummy_numbering,
    dummy_numbering,
    subdomain_material_ids);

  setup_preconditioner(precond);
  precond.build_averaging_matrix();

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_averaging_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the averaging matrix.
  ofstream out("Cd.output");
  print_sparse_matrix_to_mat(
    out, "Cd", precond.get_averaging_matrix(), 15, true, 25);
  out.close();

  out.open("dual-space-dofs-on-refined-mesh.output");
  print_weights_at_support_points_in_refined_mesh(out, precond);
  out.close();

  // Read reference data files and compare them with the generated files.
  compare_two_files("Cd.output", SOURCE_DIR "/Cd.output");
  compare_two_files("dual-space-dofs-on-refined-mesh.output",
                    SOURCE_DIR "/dual-space-dofs-on-refined-mesh.output");

  INFO("*** test end");
}
