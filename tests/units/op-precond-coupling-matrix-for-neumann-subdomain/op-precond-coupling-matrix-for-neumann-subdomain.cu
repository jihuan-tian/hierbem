/**
 * @file op-precond-coupling-matrix-for-neumann-subdomain.cu
 * @brief Verify building the coupling matrix for operator preconditioning on a
 * subdomain used in Laplace Neumann problem.
 *
 * @ingroup preconditioner
 * @author Jihuan Tian
 * @date 2025-01-27
 */

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>
#include <julia.h>

#include <fstream>
#include <set>
#include <vector>

#include "grid_out_ext.h"
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
setup_preconditioner(PreconditionerForLaplaceNeumann<2, 3, double> &precond,
                     const Triangulation<2, 3>                     &tria)
{
  precond.get_triangulation().copy_triangulation(tria);
  precond.get_triangulation().refine_global();
  precond.initialize_dof_handlers();
  precond.generate_dof_selectors();
  precond.generate_maps_between_full_and_local_dof_ids();
  precond.build_dof_to_cell_topology();
}

TEST_CASE(
  "Verify coupling matrix for operator preconditioning in Laplace Neumann on subdomain",
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
  // preconditioner's constructor. Its size is initialized to be the number of
  // nodes on the right of the interface between Dirichlet and Neumann
  // domains.
  const unsigned int n_dofs_dual_space_dual_mesh =
    (tria.n_vertices() - y_repetition - 1) / 2;
  std::vector<types::global_dof_index> dummy_numbering(
    n_dofs_dual_space_dual_mesh);
  std::set<types::material_id> subdomain_material_ids            = {2};
  std::set<types::material_id> complement_subdomain_material_ids = {1};
  PreconditionerForLaplaceNeumann<2, 3, double> precond(
    fe_primal_space,
    fe_dual_space,
    tria,
    dummy_numbering,
    dummy_numbering,
    subdomain_material_ids,
    complement_subdomain_material_ids);

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
  precond.get_coupling_matrix().print_formatted(out, 3, false, 0, "0");
  out.close();

  // Print the DoF indices and support points of the primal space on level 0,
  MappingQ<2, 3> mapping(1);
  out.open("level0-support-points.output");

  std::vector<types::global_dof_index> dof_indices_in_cell(
    fe_primal_space.dofs_per_cell);
  const std::vector<Point<2>> &unit_support_points_in_cell =
    fe_primal_space.get_unit_support_points();

  for (const auto &cell :
       precond.get_dof_handler_primal_space().mg_cell_iterators_on_level(0))
    {
      cell->get_mg_dof_indices(dof_indices_in_cell);

      // Iterate over each DoF index in the current cell.
      for (unsigned int d = 0; d < dof_indices_in_cell.size(); d++)
        {
          if (precond.get_primal_space_dof_selectors_on_primal_mesh()
                [dof_indices_in_cell[d]])
            {
              Point<3> support_point = mapping.transform_unit_to_real_cell(
                cell, unit_support_points_in_cell[d]);
              out
                << precond
                     .get_primal_space_full_to_local_dof_id_map_on_primal_mesh()
                       [dof_indices_in_cell[d]]
                << " " << support_point << "\n";
            }
        }
    }

  out.close();

  // Print the DoF indices and support points of the primal space on level 1,
  out.open("level1-support-points.output");
  for (const auto &cell :
       precond.get_dof_handler_primal_space().mg_cell_iterators_on_level(1))
    {
      cell->get_mg_dof_indices(dof_indices_in_cell);

      // Iterate over each DoF index in the current cell.
      for (unsigned int d = 0; d < dof_indices_in_cell.size(); d++)
        {
          if (precond.get_primal_space_dof_selectors_on_refined_mesh()
                [dof_indices_in_cell[d]])
            {
              Point<3> support_point = mapping.transform_unit_to_real_cell(
                cell, unit_support_points_in_cell[d]);
              out
                << precond
                     .get_primal_space_full_to_local_dof_id_map_on_refined_mesh()
                       [dof_indices_in_cell[d]]
                << " " << support_point << "\n";
            }
        }
    }

  out.close();

  (void)jl_eval_string("include(\"process.jl\")");

  jl_value_t *ret = jl_eval_string("Cp_err");
  REQUIRE(jl_unbox_float64(ret) == 0.0);

  ret = jl_eval_string("level0_support_points_err");
  REQUIRE(jl_unbox_float64(ret) == 0.0);

  ret = jl_eval_string("level1_support_points_err");
  REQUIRE(jl_unbox_float64(ret) == 0.0);

  // Finalize Julia before exit.
  jl_atexit_hook(0);
  INFO("*** test end");
}
