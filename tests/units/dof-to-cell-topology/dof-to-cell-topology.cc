/**
 * \file dof-to-cell-topology.cc
 * \brief
 * \ingroup testers
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

using namespace dealii;

template <int dim, int spacedim>
void
build_dof_to_cell_topology(
  std::vector<
    std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>>
    &dof_to_cell_topo,
  const std::vector<typename DoFHandler<dim, spacedim>::cell_iterator>
                                  &cell_iterators_in_dof_handler,
  const DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int               fe_index = 0)
{
  const types::global_dof_index        n_dofs = dof_handler.n_dofs();
  const FiniteElement<dim, spacedim>  &fe     = dof_handler.get_fe(fe_index);
  const unsigned int                   dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  dof_to_cell_topo.resize(n_dofs);

  for (const auto &cell : cell_iterators_in_dof_handler)
    {
      cell->get_dof_indices(local_dof_indices);
      for (auto dof_index : local_dof_indices)
        {
          dof_to_cell_topo[dof_index].push_back(&cell);
        }
    }
}

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

  build_dof_to_cell_topology(dof_to_cell_topo, cell_iterators, dof_handler);

  for (const auto &e : dof_to_cell_topo)
    {
      for (const typename DoFHandler<dim, spacedim>::cell_iterator *f : e)
        {
          std::cout << (*f)->active_cell_index() << std::endl;
        }
    }

  return 0;
}
