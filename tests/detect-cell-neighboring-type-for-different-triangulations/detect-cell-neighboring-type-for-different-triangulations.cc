/**
 * \file detect-cell-neighboring-type-for-different-triangulations.cc
 * \brief Get the cell neighboring type for two same or different DoF handlers
 * on two different triangulations. Then check the returned pairs of DoF indices
 * associated with common vertices.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-06-17
 */

#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

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
  auto                         surface1_to_volume_map =
    GridGenerator::extract_boundary_mesh(volume_triangulation,
                                         surface_triangulation1,
                                         surfaces);

  std::ofstream surface1_mesh_file("surface1.msh");
  GridOut().write_msh(surface_triangulation1, surface1_mesh_file);

  surfaces = {2};
  auto surface2_to_volume_map =
    GridGenerator::extract_boundary_mesh(volume_triangulation,
                                         surface_triangulation2,
                                         surfaces);

  std::ofstream surface2_mesh_file("surface2.msh");
  GridOut().write_msh(surface_triangulation2, surface2_mesh_file);

  std::cout << "=== FE_Q(2) and FE_DGQ(3) ===\n";

  {
    /**
     * Create finite element and DoF handler.
     */
    FE_Q<dim, spacedim>       fe1(2);
    FE_DGQ<dim, spacedim>     fe2(3);
    DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
    dof_handler1.initialize(surface_triangulation1, fe1);
    dof_handler2.initialize(surface_triangulation2, fe2);

    /**
     * Use a same mapping object for two cells.
     */
    const MappingQGeneric<dim, spacedim> mapping(1);

    /**
     * Check the cell neighboring type for each pair of cells.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                           common_vertex_dof_indices;
    IdeoBEM::BEMTools::CellNeighboringType cell_neighboring_type;

    unsigned int e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler1.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler2.active_cell_iterators())
          {
            cell_neighboring_type = IdeoBEM::BEMTools::
              detect_cell_neighboring_type_for_different_triangulations<
                dim,
                spacedim>(e1,
                          e2,
                          mapping,
                          mapping,
                          surface1_to_volume_map,
                          surface2_to_volume_map,
                          common_vertex_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << IdeoBEM::BEMTools::cell_neighboring_type_name(
                           cell_neighboring_type)
                      << "\n";

            if (common_vertex_dof_indices.size() > 0)
              {
                std::cout << "Vertex DoF indices in two cells: ";
                for (const auto &p : common_vertex_dof_indices)
                  {
                    std::cout << "(" << p.first << "," << p.second << "),";
                  }
                std::cout << std::endl;
              }

            e2_index++;
          }

        e1_index++;
      }

    /**
     * Save the DoF support point indices.
     */
    print_support_point_info(mapping,
                             dof_handler1,
                             "case1_fe1_order=2_support_points");
    print_support_point_info(mapping,
                             dof_handler2,
                             "case1_fe2_order=3_support_points");
  }

  std::cout << "=== FE_Q(1) and FE_DGQ(0) ===\n";

  {
    /**
     * Create finite element and DoF handler.
     */
    FE_Q<dim, spacedim>       fe1(1);
    FE_DGQ<dim, spacedim>     fe2(0);
    DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
    dof_handler1.initialize(surface_triangulation1, fe1);
    dof_handler2.initialize(surface_triangulation2, fe2);

    /**
     * Use a same mapping object for two cells.
     */
    const MappingQGeneric<dim, spacedim> mapping(1);

    /**
     * Check the cell neighboring type for each pair of cells.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                           common_vertex_dof_indices;
    IdeoBEM::BEMTools::CellNeighboringType cell_neighboring_type;

    unsigned int e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler1.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler2.active_cell_iterators())
          {
            cell_neighboring_type = IdeoBEM::BEMTools::
              detect_cell_neighboring_type_for_different_triangulations<
                dim,
                spacedim>(e1,
                          e2,
                          mapping,
                          mapping,
                          surface1_to_volume_map,
                          surface2_to_volume_map,
                          common_vertex_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << IdeoBEM::BEMTools::cell_neighboring_type_name(
                           cell_neighboring_type)
                      << "\n";

            if (common_vertex_dof_indices.size() > 0)
              {
                std::cout << "Vertex DoF indices in two cells: ";
                for (const auto &p : common_vertex_dof_indices)
                  {
                    std::cout << "(" << p.first << "," << p.second << "),";
                  }
                std::cout << std::endl;
              }

            e2_index++;
          }

        e1_index++;
      }

    /**
     * Save the DoF support point indices.
     */
    print_support_point_info(mapping,
                             dof_handler1,
                             "case2_fe1_order=1_support_points");
    print_support_point_info(mapping,
                             dof_handler2,
                             "case2_fe2_order=0_support_points");
  }

  std::cout << "=== FE_DGQ(0) and FE_DGQ(0) ===\n";

  {
    /**
     * Create finite element and DoF handler.
     */
    FE_DGQ<dim, spacedim>     fe1(0);
    FE_DGQ<dim, spacedim>     fe2(0);
    DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
    dof_handler1.initialize(surface_triangulation1, fe1);
    dof_handler2.initialize(surface_triangulation2, fe2);

    /**
     * Use a same mapping object for two cells.
     */
    const MappingQGeneric<dim, spacedim> mapping(1);

    /**
     * Check the cell neighboring type for each pair of cells.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                           common_vertex_dof_indices;
    IdeoBEM::BEMTools::CellNeighboringType cell_neighboring_type;

    unsigned int e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler1.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler2.active_cell_iterators())
          {
            cell_neighboring_type = IdeoBEM::BEMTools::
              detect_cell_neighboring_type_for_different_triangulations<
                dim,
                spacedim>(e1,
                          e2,
                          mapping,
                          mapping,
                          surface1_to_volume_map,
                          surface2_to_volume_map,
                          common_vertex_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << IdeoBEM::BEMTools::cell_neighboring_type_name(
                           cell_neighboring_type)
                      << "\n";

            if (common_vertex_dof_indices.size() > 0)
              {
                std::cout << "Vertex DoF indices in two cells: ";
                for (const auto &p : common_vertex_dof_indices)
                  {
                    std::cout << "(" << p.first << "," << p.second << "),";
                  }
                std::cout << std::endl;
              }

            e2_index++;
          }

        e1_index++;
      }

    /**
     * Save the DoF support point indices.
     */
    print_support_point_info(mapping,
                             dof_handler1,
                             "case3_fe1_order=0_support_points");
    print_support_point_info(mapping,
                             dof_handler2,
                             "case3_fe2_order=0_support_points");
  }

  return 0;
}
