/**
 * @file op-precond-coupling-matrix-for-dirichlet-subdomain.cc
 * @brief Verify building the coupling matrix for operator preconditioning on a
 * subdomain used in Laplace Dirichlet problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2025-12-24
 */

#include <deal.II/base/exceptions.h>
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

#include "debug_tools.h"
#include "grid_out_ext.h"
#include "preconditioners/preconditioner_for_laplace_dirichlet.h"

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
setup_preconditioner(PreconditionerForLaplaceDirichlet<2, 3, double> &precond,
                     const Triangulation<2, 3>                       &tria)
{
  precond.get_triangulation().copy_triangulation(tria);
  precond.get_triangulation().refine_global();
  precond.initialize_dof_handlers();
  precond.generate_dof_selectors();
  precond.generate_maps_between_full_and_local_dof_ids();
  precond.build_dof_to_cell_topology();
}

/**
 * @brief Print DoF support points in each cell in the primal mesh, then follow
 * the support points in each of its child cell in the refined mesh.
 *
 * The double level multigrid and DoF handlers are stored in the given
 * preconditioner.
 *
 * Even though the effective DoFs are confined to the subdomain, here we still
 * iterate overall cells in the triangulation, just to show that we've actually
 * selected subdomain.
 *
 * @param precond
 */
void
print_support_points_in_primal_and_refined_meshes(
  ostream                                               &out,
  const PreconditionerForLaplaceDirichlet<2, 3, double> &precond)
{
  out << "# DoF index, Support point coordinates x, y, z, Weight for basis"
      << std::endl;

  const DoFHandler<2, 3>  &dof_handler = precond.get_dof_handler_primal_space();
  const std::vector<bool> &dof_selectors_primal_mesh =
    precond.get_primal_space_dof_selectors_on_primal_mesh();
  const std::vector<bool> &dof_selectors_refined_mesh =
    precond.get_primal_space_dof_selectors_on_refined_mesh();
  std::vector<types::global_dof_index> dof_indices_in_cell(
    dof_handler.get_fe().dofs_per_cell);
  MappingQ<2, 3>               mapping(1);
  const std::vector<Point<2>> &unit_support_points =
    dof_handler.get_fe().get_unit_support_points();
  Point<2> unit_support_point = unit_support_points[0];

  // Iterate over each cell in the primal mesh.
  types::global_dof_index dof_index_primal_mesh;
  types::global_dof_index dof_index_refined_mesh;
  types::global_dof_index local_dof_index_primal_mesh;
  types::global_dof_index local_dof_index_refined_mesh;
  for (const auto &cell : dof_handler.mg_cell_iterators_on_level(0))
    {
      cell->get_mg_dof_indices(dof_indices_in_cell);
      dof_index_primal_mesh = dof_indices_in_cell[0];
      local_dof_index_primal_mesh =
        precond.get_primal_space_full_to_local_dof_id_map_on_primal_mesh()
          [dof_index_primal_mesh];
      if (dof_selectors_primal_mesh.at(dof_index_primal_mesh))
        {
          Point<3> support_point =
            mapping.transform_unit_to_real_cell(cell, unit_support_point);

          out << dof_index_primal_mesh << " " << support_point(0) << " "
              << support_point(1) << " " << support_point(2) << " " << 0
              << std::endl;
        }

      // Iterate over each child iterator of the current cell, i.e. on
      // level 1. N.B. Here we intentionally do not place the inner loop within
      // previous DoF selection predicate, since we want to verify when the DoF
      // on the primal mesh is not selected, those in refined mesh are neither
      // selected.
      for (const auto &child : cell->child_iterators())
        {
          child->get_mg_dof_indices(dof_indices_in_cell);
          dof_index_refined_mesh = dof_indices_in_cell[0];
          local_dof_index_refined_mesh =
            precond.get_primal_space_full_to_local_dof_id_map_on_refined_mesh()
              [dof_index_refined_mesh];

          // The DoFs on the primal mesh and refined mesh should be
          // simultaneously selected or not selected.
          REQUIRE(dof_selectors_primal_mesh.at(dof_index_primal_mesh) ==
                  dof_selectors_refined_mesh.at(dof_index_refined_mesh));

          if (dof_selectors_refined_mesh.at(dof_index_refined_mesh))
            {
              Point<3> support_point =
                mapping.transform_unit_to_real_cell(child, unit_support_point);

              out << dof_index_refined_mesh << " " << support_point(0) << " "
                  << support_point(1) << " " << support_point(2) << " "
                  << precond.get_coupling_matrix().el(
                       local_dof_index_primal_mesh,
                       local_dof_index_refined_mesh)
                  << std::endl;
            }
        }
    }
}

void
compare_two_files(const string file1, const string file2)
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
  "Verify coupling matrix for operator preconditioning in Laplace Dirichlet on subdomain",
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

  // Create the preconditioner. Since we do not apply the preconditioner to the
  // system matrix in this case, the conversion between internal and external
  // DoF numberings is not needed. Therefore, we pass a dummy numbering to the
  // preconditioner's constructor.
  std::vector<types::global_dof_index> dummy_numbering(tria.n_cells() / 2);
  std::set<types::material_id>         subdomain_material_ids = {1};
  PreconditionerForLaplaceDirichlet<2, 3, double> precond(
    fe_primal_space,
    fe_dual_space,
    tria,
    dummy_numbering,
    dummy_numbering,
    subdomain_material_ids);

  setup_preconditioner(precond, tria);
  precond.build_coupling_matrix();

  // Export the refined mesh.
  ofstream refined_mesh_out("refined-mesh.msh");
  write_msh_correct(precond.get_triangulation(), refined_mesh_out);

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_coupling_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the coupling matrix.
  ofstream out("Cp.output");
  print_sparse_matrix_to_mat(
    out, "Cp", precond.get_coupling_matrix(), 15, true, 25);
  out.close();

  // Print DoF indices and support point coordinates for cells in the primal
  // mesh. Following each cell data, there are DoF indices and support point
  // coordinates for the four child cells in the refined mesh.
  out.open("primal-space-dofs-on-multigrid.output");
  print_support_points_in_primal_and_refined_meshes(out, precond);
  out.close();

  // Read reference data files and compare them with the generated files.
  compare_two_files("Cp.output", SOURCE_DIR "/Cp.output");
  compare_two_files("primal-space-dofs-on-multigrid.output",
                    SOURCE_DIR "/primal-space-dofs-on-multigrid.output");

  INFO("*** test end");
}
