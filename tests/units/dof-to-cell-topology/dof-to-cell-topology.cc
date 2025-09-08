// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file dof-to-cell-topology.cc
 * \brief
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-03-02
 */

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <vector>

#inlcude "dofs/dof_tools_ext.h"

using namespace HierBEM;
using namespace dealii;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(1);
  //  GridOut       grid_out;
  //  std::ofstream mesh_file("grid.msh");
  //  grid_out.write_msh(triangulation, mesh_file);

  DoFHandler<dim, spacedim> dof_handler(triangulation);
  FE_Q<dim, spacedim>       fe(2);
  dof_handler.distribute_dofs(fe);

  const unsigned int n_cells = dof_handler.get_triangulation().n_active_cells();
  std::cout << "Number of cells: " << n_cells << std::endl;

  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators;
  cell_iterators.reserve(triangulation.n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_iterators.push_back(cell);
    }

  std::vector<
    std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>>
    dof_to_cell_topo;

  DoFToolsExt::build_dof_to_cell_topology(dof_to_cell_topo,
                                          cell_iterators,
                                          dof_handler);

  for (const auto &e : dof_to_cell_topo)
    {
      for (const typename DoFHandler<dim, spacedim>::cell_iterator *f : e)
        {
          std::cout << (*f)->active_cell_index() << std::endl;
        }
    }

  return 0;
}
