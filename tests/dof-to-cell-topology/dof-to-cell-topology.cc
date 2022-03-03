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
#include <vector>

#include "sauter_quadrature.h"

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

  std::vector<std::vector<unsigned int>> dof_to_cell_topo;

  build_dof_to_cell_topology(dof_to_cell_topo, dof_handler);

  print_dof_to_cell_topology(dof_to_cell_topo);

  return 0;
}
