/**
 * \file detect-cell-neighboring-type-for-same-triangulations.cc
 * \brief Get the cell neighboring type for two different DoF handlers on a same
 * triangulation, i.e. the encompassed finite elements in the two DoF handlers
 * are different. Then check the returned pairs of DoF indices associated with
 * common vertices.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-06-11
 */

#include <deal.II/base/logstream.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "bem/bem_tools.h"
#include "utilities/debug_tools.h"

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

  std::cout << "=== FE_Q(1) and FE_DGQ(2) ===\n";

  {
    /**
     * Create finite element and DoF handler.
     */
    FE_Q<dim, spacedim>       fe1(1);
    FE_DGQ<dim, spacedim>     fe2(2);
    DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
    dof_handler1.initialize(surface_triangulation, fe1);
    dof_handler2.initialize(surface_triangulation, fe2);

    /**
     * Use a same mapping object for two cells.
     */
    const MappingQ<dim, spacedim> mapping(1);

    /**
     * Check the cell neighboring type for each pair of cells.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                           common_vertex_dof_indices;
    HierBEM::BEMTools::CellNeighboringType cell_neighboring_type;
    unsigned int                           e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler1.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler2.active_cell_iterators())
          {
            cell_neighboring_type = HierBEM::BEMTools::
              detect_cell_neighboring_type_for_same_triangulations(
                e1, e2, mapping, mapping, common_vertex_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << HierBEM::BEMTools::cell_neighboring_type_name(
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
                             "case1_fe1_order=1_support_points");
    print_support_point_info(mapping,
                             dof_handler2,
                             "case1_fe2_order=2_support_points");
  }

  std::cout << "=== FE_Q(1) and FE_DGQ(0) ===\n";

  {
    /**
     * Create finite element and DoF handler.
     */
    FE_Q<dim, spacedim>       fe1(1);
    FE_DGQ<dim, spacedim>     fe2(0);
    DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
    dof_handler1.initialize(surface_triangulation, fe1);
    dof_handler2.initialize(surface_triangulation, fe2);

    /**
     * Use a same mapping object for two cells.
     */
    const MappingQ<dim, spacedim> mapping(1);

    /**
     * Check the cell neighboring type for each pair of cells.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                           common_vertex_dof_indices;
    HierBEM::BEMTools::CellNeighboringType cell_neighboring_type;
    unsigned int                           e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler1.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler2.active_cell_iterators())
          {
            cell_neighboring_type = HierBEM::BEMTools::
              detect_cell_neighboring_type_for_same_triangulations(
                e1, e2, mapping, mapping, common_vertex_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << HierBEM::BEMTools::cell_neighboring_type_name(
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
    dof_handler1.initialize(surface_triangulation, fe1);
    dof_handler2.initialize(surface_triangulation, fe2);

    /**
     * Use a same mapping object for two cells.
     */
    const MappingQ<dim, spacedim> mapping(1);

    /**
     * Check the cell neighboring type for each pair of cells.
     */
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
                                           common_vertex_dof_indices;
    HierBEM::BEMTools::CellNeighboringType cell_neighboring_type;
    unsigned int                           e1_index = 0;
    for (const typename DoFHandler<dim, spacedim>::active_cell_iterator &e1 :
         dof_handler1.active_cell_iterators())
      {
        unsigned int e2_index = 0;
        for (const typename DoFHandler<dim, spacedim>::active_cell_iterator
               &e2 : dof_handler2.active_cell_iterators())
          {
            cell_neighboring_type = HierBEM::BEMTools::
              detect_cell_neighboring_type_for_same_triangulations(
                e1, e2, mapping, mapping, common_vertex_dof_indices);

            std::cout << "Cell neighboring type for cells (" << e1_index << ","
                      << e2_index << ") is: "
                      << HierBEM::BEMTools::cell_neighboring_type_name(
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
