/**
 * @file op-precond-averaging-matrix-for-neumann.cu
 * @brief Verify building the averaging matrix for operator preconditioning used
 * in Laplace Neumann problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2025-01-08
 */
#include <deal.II/base/point.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>
#include <julia.h>

#include <fstream>
#include <vector>

#include "grid_out_ext.h"
#include "preconditioners/preconditioner_for_laplace_neumann.h"

using namespace Catch::Matchers;
using namespace HierBEM;
using namespace dealii;
using namespace std;

TEST_CASE(
  "Verify averaging matrix for operator preconditioning in Laplace Neumann",
  "[preconditioner]")
{
  INFO("*** test start");
  // Start Julia for postprocessing.
  jl_init();


  // Define the primal space and dual space with respect to the hyper singular
  // operator.
  FE_Q<2, 3>   fe_primal_space(1);
  FE_DGQ<2, 3> fe_dual_space(0);

  REQUIRE(fe_dual_space.has_support_points());

  // Generate the mesh. Because we are going to distribute DoFs on the
  // two-level multigrid required by the operator preconditioner, the
  // triangulation object should be constructed with a level difference
  // limitation at vertices.
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
  std::vector<types::global_dof_index>          dummy_numbering;
  PreconditionerForLaplaceNeumann<2, 3, double> precond(
    fe_primal_space, fe_dual_space, tria, dummy_numbering, dummy_numbering);

  // Build the averaging matrix.
  precond.initialize_dof_handlers();
  precond.build_dof_to_cell_topology();
  precond.build_averaging_matrix();

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_averaging_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the averaging matrix.
  ofstream out("Cd.output");
  precond.get_averaging_matrix().print_formatted(out, 3, false, 0, "0");
  out.close();

  // Print the DoF indices and support points of the primal space on level 0,
  // which is equivalent to DoFs in the dual space on the dual mesh.
  MappingQ<2, 3> mapping(1);
  out.open("support-points-in-dual-space-on-dual-mesh.output");

  std::vector<types::global_dof_index> dof_indices_in_primal_cell(
    fe_primal_space.dofs_per_cell);
  const std::vector<Point<2>> &unit_support_points_in_primal_cell =
    fe_primal_space.get_unit_support_points();

  for (const auto &cell :
       precond.get_dof_handler_primal_space().mg_cell_iterators_on_level(0))
    {
      cell->get_mg_dof_indices(dof_indices_in_primal_cell);

      // Iterate over each DoF index in the current cell.
      for (unsigned int d = 0; d < dof_indices_in_primal_cell.size(); d++)
        {
          Point<3> support_point = mapping.transform_unit_to_real_cell(
            cell, unit_support_points_in_primal_cell[d]);
          out << dof_indices_in_primal_cell[d] << " " << support_point << "\n";
        }
    }

  out.close();

  // Print the DoF indices and support points of the dual space on the refined
  // mesh.
  out.open("support-points-in-dual-space-on-refined-mesh.output");
  std::vector<types::global_dof_index> dof_indices_in_refined_cell(
    fe_dual_space.dofs_per_cell);
  const std::vector<Point<2>> &unit_support_points_in_refined_cell =
    fe_dual_space.get_unit_support_points();

  for (const auto &cell :
       precond.get_dof_handler_dual_space().mg_cell_iterators_on_level(1))
    {
      cell->get_mg_dof_indices(dof_indices_in_refined_cell);

      // Iterate over each DoF index in the current cell.
      for (unsigned int d = 0; d < dof_indices_in_refined_cell.size(); d++)
        {
          Point<3> support_point = mapping.transform_unit_to_real_cell(
            cell, unit_support_points_in_refined_cell[d]);
          out << dof_indices_in_refined_cell[d] << " " << support_point << "\n";
        }
    }

  out.close();

  (void)jl_eval_string("include(\"process.jl\")");

  jl_value_t *ret = jl_eval_string("Cd_err");
  REQUIRE(jl_unbox_float64(ret) == 0.0);

  ret = jl_eval_string("support_points_in_dual_mesh_err");
  REQUIRE(jl_unbox_float64(ret) == 0.0);

  ret = jl_eval_string("support_points_in_refined_mesh_err");
  REQUIRE(jl_unbox_float64(ret) == 0.0);

  // Finalize Julia before exit.
  jl_atexit_hook(0);
  INFO("*** test end");
}
