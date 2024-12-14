/**
 * \file detect-cell-neighboring-type-for-same-dofhandlers.cc
 * \brief Get the cell neighboring type for two same DoF handlers and check the
 * returned pairs of DoF indices associated with common vertices.
 *
 * \ingroup testers
 * \author Jihuan Tian
 * \date 2022-06-10
 */

#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "bem_tools.hcu"
#include "debug_tools.h"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  /**
   * Generate a triangulation containing only one cell.
   */
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  Triangulation<spacedim, spacedim> volume_triangulation;
  std::vector<unsigned int>         repetitions{1, 1, 1};
  GridGenerator::subdivided_hyper_rectangle(volume_triangulation,
                                            repetitions,
                                            Point<spacedim>(0, 0, 0),
                                            Point<spacedim>(1, 1, 1));

  std::ofstream mesh_file("single-cell.msh");
  GridOut().write_msh(volume_triangulation, mesh_file);
  mesh_file.close();

  /**
   * Extract the boundary mesh.
   */
  Triangulation<dim, spacedim> surface_triangulation;
  GridGenerator::extract_boundary_mesh(volume_triangulation,
                                       surface_triangulation);

  /**
   * Save the boundary mesh.
   */
  mesh_file.open("single-cell-boundary.msh");
  GridOut().write_msh(surface_triangulation, mesh_file);
  mesh_file.close();

  /**
   * Use a same mapping object for two cells.
   */
  const MappingQGeneric<dim, spacedim> mapping(1);

  /**
   * Check the cell neighboring type for each pair of cells.
   */
  std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                         common_vertex_pair_dof_indices;
  HierBEM::BEMTools::CellNeighboringType cell_neighboring_type;

  {
    std::cout << "=== FE_Q(2) ===" << std::endl;
    /**
     * Create finite element and DoF handler.
     */
    FE_Q<dim, spacedim>       fe(2);
    DoFHandler<dim, spacedim> dof_handler;
    dof_handler.initialize(surface_triangulation, fe);

    unsigned int e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler.active_cell_iterators())
          {
            cell_neighboring_type = HierBEM::BEMTools::
              detect_cell_neighboring_type_for_same_dofhandlers(
                e1, e2, mapping, mapping, common_vertex_pair_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << HierBEM::BEMTools::cell_neighboring_type_name(
                           cell_neighboring_type)
                      << "\n";

            if (common_vertex_pair_dof_indices.size() > 0)
              {
                std::cout << "Common vertex DoF indices: ";
                for (const auto &dof_index_pair :
                     common_vertex_pair_dof_indices)
                  {
                    std::cout << "(" << dof_index_pair.first << ","
                              << dof_index_pair.second << ") ";
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
    print_support_point_info(mapping, dof_handler, "fe_q_support_points");
  }

  {
    std::cout << "=== FE_DGQ(2) ===" << std::endl;
    /**
     * Create finite element and DoF handler.
     */
    FE_DGQ<dim, spacedim>     fe(2);
    DoFHandler<dim, spacedim> dof_handler;
    dof_handler.initialize(surface_triangulation, fe);

    unsigned int e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler.active_cell_iterators())
          {
            cell_neighboring_type = HierBEM::BEMTools::
              detect_cell_neighboring_type_for_same_dofhandlers<dim, spacedim>(
                e1, e2, mapping, mapping, common_vertex_pair_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << HierBEM::BEMTools::cell_neighboring_type_name(
                           cell_neighboring_type)
                      << "\n";

            if (common_vertex_pair_dof_indices.size() > 0)
              {
                std::cout << "Common vertex DoF indices: ";
                for (const auto &dof_index_pair :
                     common_vertex_pair_dof_indices)
                  {
                    std::cout << "(" << dof_index_pair.first << ","
                              << dof_index_pair.second << ") ";
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
    print_support_point_info(mapping, dof_handler, "fe_dgq_support_points");
  }

  {
    std::cout << "=== FE_DGQ(0) ===" << std::endl;
    /**
     * Create finite element and DoF handler.
     */
    FE_DGQ<dim, spacedim>     fe(0);
    DoFHandler<dim, spacedim> dof_handler;
    dof_handler.initialize(surface_triangulation, fe);

    unsigned int e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler.active_cell_iterators())
          {
            cell_neighboring_type = HierBEM::BEMTools::
              detect_cell_neighboring_type_for_same_dofhandlers<dim, spacedim>(
                e1, e2, mapping, mapping, common_vertex_pair_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << HierBEM::BEMTools::cell_neighboring_type_name(
                           cell_neighboring_type)
                      << "\n";

            if (common_vertex_pair_dof_indices.size() > 0)
              {
                std::cout << "Common vertex DoF indices: ";
                for (const auto &dof_index_pair :
                     common_vertex_pair_dof_indices)
                  {
                    std::cout << "(" << dof_index_pair.first << ","
                              << dof_index_pair.second << ") ";
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
    print_support_point_info(mapping, dof_handler, "fe_dgq_support_points");
  }

  return 0;
}
