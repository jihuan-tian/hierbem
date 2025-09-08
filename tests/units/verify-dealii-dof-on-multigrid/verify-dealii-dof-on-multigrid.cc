// Copyright (C) 2024 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-dealii-dof-on-multigrid.cc
 * @brief Verify the DoF numbering on multigrid in deal.ii
 *
 * Conclusion: DoFs on different mesh levels are independently numbered.
 *
 * @ingroup dealii_verify
 * @author Jihuan Tian
 * @date 2024-12-02
 */

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace dealii;

int
main(int argc, const char *argv[])
{
  (void)argc;
  (void)argv;

  // Generate a 2x2 grid and perform a global refinement. In order to distribute
  // DoFs in a DoF handler to multigrid, the level difference at vertices should
  // be limited. This can be passed as an argument in the constructor of
  // the Triangulation class.
  Triangulation<2, 3> tria(
    Triangulation<2, 3>::MeshSmoothing::limit_level_difference_at_vertices);
  GridGenerator::subdivided_hyper_cube(tria, 2, 0, 4);
  tria.refine_global();
  ofstream mesh_out("mesh.msh");
  GridOut  grid_out;
  grid_out.write_msh(tria, mesh_out);

  // Create DoF handler and finite element. Then distribute DoFs over the
  // multigrid.
  DoFHandler<2, 3> dof_handler(tria);
  FE_Q<2, 3>       fe(2);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  // Print the DoF information.
  cout << "Number of active DoFs: " << dof_handler.n_dofs() << "\n";
  cout << "Number of DoFs on level 0: " << dof_handler.n_dofs(0) << "\n";
  cout << "Number of DoFs on level 1: " << dof_handler.n_dofs(1) << endl;
  // N.B. Because we've performed once global refinement, the number of active
  // DoFs should be equal to the number of DoFs on level 1.
  AssertDimension(dof_handler.n_dofs(), dof_handler.n_dofs(1));

  // Print out DoF indices on level 0.
  cout << "=== Cellwise DoF indices on level 0 ===\n";
  vector<types::global_dof_index> dof_indices_in_cell(fe.dofs_per_cell);
  unsigned int                    i = 0;
  for (const auto &cell : dof_handler.mg_cell_iterators_on_level(0))
    {
      // get_dof_indices should not be used, because it only works on active
      // cells.
      cell->get_mg_dof_indices(dof_indices_in_cell);
      cout << "Cell #" << i << ": ";
      for (auto index : dof_indices_in_cell)
        {
          cout << index << " ";
        }
      cout << endl;
      i++;
    }

  cout << "=== Cellwise DoF indices on level 1 ===\n";
  i = 0;
  for (const auto &cell : dof_handler.mg_cell_iterators_on_level(1))
    {
      // get_dof_indices should not be used, because it only works on active
      // cells.
      cell->get_mg_dof_indices(dof_indices_in_cell);
      cout << "Cell #" << i << ": ";
      for (auto index : dof_indices_in_cell)
        {
          cout << index << " ";
        }
      cout << endl;
      i++;
    }

  cout << "=== Cellwise vertex DoF indices on level 0 ===\n";
  i = 0;
  for (const auto &cell : dof_handler.mg_cell_iterators_on_level(0))
    {
      cout << "Cell #" << i << ": ";
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; v++)
        {
          cout << cell->mg_vertex_dof_index(0, v, 0) << " ";
        }
      cout << endl;
      i++;
    }

  cout << "=== Cellwise vertex DoF indices on level 1 ===\n";
  i = 0;
  for (const auto &cell : dof_handler.mg_cell_iterators_on_level(1))
    {
      cout << "Cell #" << i << ": ";
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; v++)
        {
          cout << cell->mg_vertex_dof_index(1, v, 0) << " ";
        }
      cout << endl;
      i++;
    }

  return 0;
}
