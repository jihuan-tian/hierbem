/**
 * @file op-precond-mass-matrix-for-neumann-subdomain.cu
 * @brief Verify building the mass matrix for operator preconditioning on a
 * subdomain used in Laplace Neumann problem.
 *
 * @ingroup preconditioners
 * @author Jihuan Tian
 * @date 2025-01-29
 */

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "grid/grid_out_ext.h"
#include "preconditioners/preconditioner_for_laplace_neumann.h"

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
  PreconditionerForLaplaceNeumann<2, 3, double, double> &precond)
{
  precond.initialize_dof_handlers();
  precond.generate_dof_selectors();
  precond.generate_maps_between_full_and_local_dof_ids();
  precond.build_dof_to_cell_topology();
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
  "Verify mass matrix for operator preconditioning in Laplace Neumann on subdomain",
  "[preconditioner]")
{
  INFO("*** test start");

  // Define the primal space and dual space with respect to the hyper singular
  // operator.
  FE_Q<2, 3>   fe_primal_space(1);
  FE_DGQ<2, 3> fe_dual_space(0);

  REQUIRE(fe_dual_space.has_support_points());

  // Generate the mesh. Because we are going to distribute DoFs on the
  // two-level multigrid required by the operator preconditioner, the
  // triangulation object should be constructed with a level difference
  // limitation at vertices.
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
  const unsigned int n_vertices = tria.n_vertices();
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
  // preconditioner's constructor. Its size is initialized to be the number of
  // nodes on the right of the interface between Dirichlet and Neumann domains.
  const unsigned int n_dofs_dual_space_dual_mesh =
    (n_vertices - y_repetition - 1) / 2;
  std::vector<types::global_dof_index> dummy_numbering(
    n_dofs_dual_space_dual_mesh);
  std::set<types::material_id> subdomain_material_ids            = {2};
  std::set<types::material_id> complement_subdomain_material_ids = {1};
  PreconditionerForLaplaceNeumann<2, 3, double, double> precond(
    fe_primal_space,
    fe_dual_space,
    tria,
    dummy_numbering,
    dummy_numbering,
    subdomain_material_ids,
    complement_subdomain_material_ids);

  setup_preconditioner(precond);
  precond.build_coupling_matrix();
  precond.build_averaging_matrix();
  precond.build_mass_matrix_on_refined_mesh(QGauss<2>(2));

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_mass_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the mass matrix.
  ofstream out("Mr.output");
  precond.get_mass_matrix().print_formatted(out, 15, false, 25, "0");
  out.close();

  REQUIRE(precond.get_averaging_matrix().n() == precond.get_mass_matrix().m());
  REQUIRE(precond.get_coupling_matrix().n() == precond.get_mass_matrix().n());

  // Read reference data files and compare them with the generated files.
  compare_two_files("Mr.output", SOURCE_DIR "/Mr.output");

  INFO("*** test end");
}
