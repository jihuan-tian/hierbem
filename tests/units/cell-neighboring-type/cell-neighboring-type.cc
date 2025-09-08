// Copyright (C) 2021-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * \file cell-neighboring-type.cc
 * \brief Verify the detection of cell neighboring type.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2020-11-10
 */

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/full_matrix.templates.h>

#include <fstream>
#include <iostream>

#include "laplace/laplace_bem.h"

int
main()
{
  deallog.pop();
  deallog.depth_console(2);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  // Generate a grid.
  Triangulation<dim, spacedim> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  //  GridOut gridout;
  //  std::ofstream mesh_file("./cell-neighboring-type.msh");
  //  gridout.write_msh(triangulation, mesh_file);

  // Calculate cell neighboring type using vertex indices.
  const unsigned int       n_active_cells = triangulation.n_active_cells();
  FullMatrix<unsigned int> cell_neighboring_type_matrix(n_active_cells,
                                                        n_active_cells);
  // Iterative over each pair of cells.
  types::global_vertex_index i = 0;

  for (const auto first_cell : triangulation.active_cell_iterators())
    {
      std::array<types::global_vertex_index,
                 GeometryInfo<dim>::vertices_per_cell>
        first_cell_vertex_indices(
          HierBEM::get_vertex_indices<dim, spacedim>(first_cell));

      types::global_vertex_index j = 0;
      for (const auto second_cell : triangulation.active_cell_iterators())
        {
          std::array<types::global_vertex_index,
                     GeometryInfo<dim>::vertices_per_cell>
            second_cell_vertex_indices(
              HierBEM::get_vertex_indices<dim, spacedim>(second_cell));

          std::vector<types::global_vertex_index> vertex_index_intersection;
          vertex_index_intersection.reserve(
            GeometryInfo<dim>::vertices_per_cell);
          cell_neighboring_type_matrix(i, j) =
            HierBEM::detect_cell_neighboring_type<dim>(
              first_cell_vertex_indices,
              second_cell_vertex_indices,
              vertex_index_intersection);

          j++;
        }

      i++;
    }

  deallog << "Calculate cell neighboring type using vertex indices..."
          << std::endl;
  cell_neighboring_type_matrix.print(deallog);

  // Calculate cell neighboring type using vertex dof indices.
  FE_Q<dim, spacedim>       fe(2);
  DoFHandler<dim, spacedim> dof_handler;
  dof_handler.initialize(triangulation, fe);

  cell_neighboring_type_matrix = 0.;
  i                            = 0;
  for (const auto first_cell : dof_handler.active_cell_iterators())
    {
      std::array<types::global_dof_index, GeometryInfo<dim>::vertices_per_cell>
        first_cell_vertex_dof_indices(
          HierBEM::get_vertex_dof_indices<dim, spacedim>(first_cell));

      types::global_vertex_index j = 0;
      for (const auto second_cell : dof_handler.active_cell_iterators())
        {
          std::array<types::global_dof_index,
                     GeometryInfo<dim>::vertices_per_cell>
            second_cell_vertex_dof_indices(
              HierBEM::get_vertex_dof_indices<dim, spacedim>(second_cell));

          std::vector<types::global_dof_index> vertex_dof_index_intersection;
          vertex_dof_index_intersection.reserve(
            GeometryInfo<dim>::vertices_per_cell);
          cell_neighboring_type_matrix(i, j) =
            HierBEM::detect_cell_neighboring_type<dim>(
              first_cell_vertex_dof_indices,
              second_cell_vertex_dof_indices,
              vertex_dof_index_intersection);

          j++;
        }

      i++;
    }

  deallog << "Calculate cell neighboring type using vertex dof indices..."
          << std::endl;
  cell_neighboring_type_matrix.print(deallog);

  return 0;
}
