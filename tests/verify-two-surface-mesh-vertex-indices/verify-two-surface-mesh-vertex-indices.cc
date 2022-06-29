/**
 * \file verify-two-surface-mesh-vertex-indices.cc
 * \brief Verify vertex indices for two neighboring boundary triangulations and
 * check if they are renumbered in each boundary triangulation. If so, there
 * should be additional work to create a mapping relation between between the
 * two triangulations for those shared vertices.
 *
 * \ingroup testers dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-06
 */

#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "bem_tools.h"
#include "debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  std::ifstream                     mesh("neighboring-squares_hex.msh");
  Triangulation<spacedim, spacedim> volume_triangulation;
  GridIn<spacedim, spacedim>        grid_in;
  grid_in.attach_triangulation(volume_triangulation);
  grid_in.read_msh(mesh);

  std::set<types::boundary_id> surfaces{1};
  Triangulation<dim, spacedim> surface_triangulation1, surface_triangulation2;
  GridGenerator::extract_boundary_mesh(volume_triangulation,
                                       surface_triangulation1,
                                       surfaces);

  std::ofstream surface1_mesh_file("surface1.msh");
  GridOut().write_msh(surface_triangulation1, surface1_mesh_file);

  surfaces = {2};
  GridGenerator::extract_boundary_mesh(volume_triangulation,
                                       surface_triangulation2,
                                       surfaces);

  std::ofstream surface2_mesh_file("surface2.msh");
  GridOut().write_msh(surface_triangulation2, surface2_mesh_file);

  std::cout << "=== Vertices in surface triangulation #1 ===\n";
  for (const auto &e : surface_triangulation1.active_cell_iterators())
    {
      auto vertices_in_cell =
        IdeoBEM::BEMTools::get_vertex_indices_in_cell<dim, spacedim>(e);

      print_vector_values(std::cout, vertices_in_cell, ",", false);
      std::cout << std::endl;
    }

  std::cout << "=== Vertices in surface triangulation #2 ===\n";
  for (const auto &e : surface_triangulation2.active_cell_iterators())
    {
      auto vertices_in_cell =
        IdeoBEM::BEMTools::get_vertex_indices_in_cell<dim, spacedim>(e);

      print_vector_values(std::cout, vertices_in_cell, ",", false);
      std::cout << std::endl;
    }
}
