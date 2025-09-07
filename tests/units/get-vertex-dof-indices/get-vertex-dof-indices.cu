/**
 * \file get-vertex-dof-indices.cc
 * \brief Verify the extraction of vertex DoF indices for both @p FE_Q and
 * @p FE_DGQ.
 *
 * \ingroup test_cases
 * \author Jihuan Tian
 * \date 2022-06-15
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

int
main()
{
  /**
   * General a two-cell mesh.
   */
  const unsigned int           dim      = 2;
  const unsigned int           spacedim = 3;
  Triangulation<dim, spacedim> tria;
  std::vector<unsigned int>    repetitions{2, 1};
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            repetitions,
                                            Point<dim>(0, 0),
                                            Point<dim>(2, 1));

  std::ofstream mesh_file("two-cells.msh");
  GridOut().write_msh(tria, mesh_file);
  mesh_file.close();

  MappingQ<dim, spacedim> mapping(1);

  {
    FE_Q<dim, spacedim>       fe(3);
    DoFHandler<dim, spacedim> dof_handler;
    dof_handler.initialize(tria, fe);

    for (const auto &e : dof_handler.active_cell_iterators())
      {
        print_vector_values<std::array<types::global_dof_index,
                                       GeometryInfo<dim>::vertices_per_cell>>(
          std::cout,
          HierBEM::BEMTools::get_vertex_dof_indices_in_cell(e, mapping),
          ",",
          true);
      }

    std::map<types::global_dof_index, Point<spacedim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
    std::ofstream dof_support_point_file("fe_q_support_points.gpl");
    DoFTools::write_gnuplot_dof_support_point_info(dof_support_point_file,
                                                   support_points);
  }

  {
    FE_DGQ<dim, spacedim>     fe(3);
    DoFHandler<dim, spacedim> dof_handler;
    dof_handler.initialize(tria, fe);

    for (const auto &e : dof_handler.active_cell_iterators())
      {
        print_vector_values<std::array<types::global_dof_index,
                                       GeometryInfo<dim>::vertices_per_cell>>(
          std::cout,
          HierBEM::BEMTools::get_vertex_dof_indices_in_cell(e, mapping),
          ",",
          true);
      }

    std::map<types::global_dof_index, Point<spacedim>> support_points;
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
    std::ofstream dof_support_point_file("fe_dgq_support_points.gpl");
    DoFTools::write_gnuplot_dof_support_point_info(dof_support_point_file,
                                                   support_points);
  }

  return 0;
}
