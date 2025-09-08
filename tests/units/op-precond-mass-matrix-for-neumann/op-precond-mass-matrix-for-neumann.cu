// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file op-precond-mass-matrix-for-neumann.cu
 * @brief Verify building the mass matrix for operator preconditioning used
 * in Laplace Neumann problem.
 *
 * @ingroup preconditioners
 * @author Jihuan Tian
 * @date 2025-01-29
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>

#include <fstream>
#include <string>
#include <vector>

#include "grid/grid_out_ext.h"
#include "preconditioners/preconditioner_for_laplace_neumann.h"

using namespace Catch::Matchers;
using namespace HierBEM;
using namespace dealii;
using namespace std;

void
setup_preconditioner(
  PreconditionerForLaplaceNeumann<2, 3, double, double> &precond)
{
  precond.initialize_dof_handlers();
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

TEST_CASE("Verify mass matrix for operator preconditioning in Laplace Neumann",
          "[preconditioner]")
{
  INFO("*** test start");

  // Define the primal space and dual space with respect to the hyper singular
  // operator.
  FE_Q<2, 3>   fe_primal_space(1);
  FE_DGQ<2, 3> fe_dual_space(0);

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
  std::vector<types::global_dof_index>                  dummy_numbering;
  PreconditionerForLaplaceNeumann<2, 3, double, double> precond(
    fe_primal_space, fe_dual_space, tria, dummy_numbering, dummy_numbering);

  // Setup the preconditioner and build matrices.
  setup_preconditioner(precond);
  precond.build_coupling_matrix();
  precond.build_averaging_matrix();
  precond.build_mass_matrix_on_refined_mesh(QGauss<2>(2));

  // Print the sparsity pattern.
  ofstream sp_pattern("sparsity-pattern.svg");
  precond.get_mass_matrix().get_sparsity_pattern().print_svg(sp_pattern);

  // Print the mass matrix.
  ofstream out("Mr.output");
  precond.get_mass_matrix().print_formatted(out, 15, true, 25, "0");
  out.close();

  REQUIRE(precond.get_averaging_matrix().n() == precond.get_mass_matrix().m());
  REQUIRE(precond.get_coupling_matrix().n() == precond.get_mass_matrix().n());

  // Read reference data files and compare them with the generated files.
  compare_two_files("Mr.output", SOURCE_DIR "/Mr.output");

  INFO("*** test end");
}
