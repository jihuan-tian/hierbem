/**
 * \file verify-cell-iters-in-dofhandler-and-triangulation.cc
 * \brief Verify if the cell iterator obtained from a DoF handler is the same as
 * the cell iterator obtained from a triangulation.
 *
 * \ingroup testers dealii_verify
 * \author Jihuan Tian
 * \date 2022-06-09
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "bem_tools.hcu"
#include "debug_tools.h"

using namespace dealii;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  /**
   * Generate the volume mesh.
   */
  Triangulation<spacedim, spacedim> volume_triangulation;
  GridGenerator::hyper_ball(volume_triangulation,
                            Point<spacedim>(0, 0, 0),
                            1.0,
                            true);
  volume_triangulation.refine_global(1);

  /**
   * Extract the boundary mesh.
   */
  Triangulation<dim, spacedim> surface_triangulation;
  std::map<typename Triangulation<dim, spacedim>::cell_iterator,
           typename Triangulation<spacedim, spacedim>::face_iterator>
    map_from_surface_mesh_to_volume_mesh =
      GridGenerator::extract_boundary_mesh(volume_triangulation,
                                           surface_triangulation);

  /**
   * Create two finite elements which are to be associated with a same
   * triangulation.
   */
  FE_Q<dim, spacedim>   fe_q(2);
  FE_DGQ<dim, spacedim> fe_dgq(2);

  /**
   * Create and initialize two @p DoFHandlers.
   */
  DoFHandler<dim, spacedim> dof_handler1, dof_handler2;
  dof_handler1.initialize(surface_triangulation, fe_q);
  dof_handler2.initialize(surface_triangulation, fe_dgq);

  typename DoFHandler<dim, spacedim>::cell_iterator cell_iter_for_fe_q =
    dof_handler1.begin_active();
  typename DoFHandler<dim, spacedim>::cell_iterator cell_iter_for_fe_dgq =
    dof_handler2.begin_active();

  for (; cell_iter_for_fe_q != dof_handler1.end();
       cell_iter_for_fe_q++, cell_iter_for_fe_dgq++)
    {
      /**
       * Try to find the cell iterators obtained from the two DoF handlers in
       * the map from surface mesh to volume mesh.
       */
      auto pos = map_from_surface_mesh_to_volume_mesh.find(cell_iter_for_fe_q);
      if (pos != map_from_surface_mesh_to_volume_mesh.end())
        {
          std::cout << "Cell iterator in DoFHandler1 can be found!"
                    << std::endl;
        }
      else
        {
          std::cout << "Cell iterator in DoFHandler1 cannot be found!"
                    << std::endl;
        }

      pos = map_from_surface_mesh_to_volume_mesh.find(cell_iter_for_fe_dgq);
      if (pos != map_from_surface_mesh_to_volume_mesh.end())
        {
          std::cout << "Cell iterator in DoFHandler2 can be found!"
                    << std::endl;
        }
      else
        {
          std::cout << "Cell iterator in DoFHandler2 cannot be found!"
                    << std::endl;
        }
    }

  return 0;
}
