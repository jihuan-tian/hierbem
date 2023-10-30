/**
 * @file verify-extract-boundary-mesh.cc
 * @brief Verify if the extracted boundary mesh has been correctly refined
 * according to the refined volume mesh, i.e. the extracted surface mesh should
 * conforms to the manifold.
 *
 * @ingroup testers dealii_verify
 * @author Jihuan Tian
 * @date 2023-10-29
 */

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>

#include "grid_out_ext.h"

using namespace dealii;
using namespace std;
using namespace HierBEM;

int
main()
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  Triangulation<spacedim> left_ball, right_ball, tria;
  double                  inter_distance = 8;
  double                  radius         = 1.0;

  GridGenerator::hyper_ball(left_ball,
                            Point<spacedim>(-inter_distance / 2.0, 0, 0),
                            radius);
  GridGenerator::hyper_ball(right_ball,
                            Point<spacedim>(inter_distance / 2.0, 0, 0),
                            radius);

  /**
   * @internal Set different manifold ids and material ids to all the cells in
   * the two balls.
   */
  for (typename Triangulation<spacedim>::active_cell_iterator cell =
         left_ball.begin_active();
       cell != left_ball.end();
       cell++)
    {
      cell->set_all_manifold_ids(0);
      cell->set_material_id(0);
    }

  for (typename Triangulation<spacedim>::active_cell_iterator cell =
         right_ball.begin_active();
       cell != right_ball.end();
       cell++)
    {
      cell->set_all_manifold_ids(1);
      cell->set_material_id(1);
    }

  /**
   * @internal @p merge_triangulation can only operate on coarse mesh, i.e.
   * triangulations not refined. During the merging, the material ids are
   * copied. When the last argument is true, the manifold ids are copied.
   * Boundary ids will not be copied.
   */
  GridGenerator::merge_triangulations(left_ball, right_ball, tria, 1e-12, true);

  /**
   * @internal Assign manifold objects to the two balls in the merged mesh.
   */
  const SphericalManifold<spacedim> left_ball_manifold(
    Point<spacedim>(-inter_distance / 2.0, 0, 0));
  const SphericalManifold<spacedim> right_ball_manifold(
    Point<spacedim>(inter_distance / 2.0, 0, 0));

  tria.set_manifold(0, left_ball_manifold);
  tria.set_manifold(1, right_ball_manifold);

  // Refine the volume mesh.
  tria.refine_global(1);

  // Extract the boundary mesh. N.B. Before the operation, the association of
  // manifold objects and manifold ids must also be set for the surface
  // triangulation. The manifold objects for the surface triangulation have
  // different dimension template paramreters as those for the volume
  // triangulation.
  Triangulation<dim, spacedim> surface_tria;

  const SphericalManifold<dim, spacedim> left_ball_surface_manifold(
    Point<spacedim>(-inter_distance / 2.0, 0, 0));
  const SphericalManifold<dim, spacedim> right_ball_surface_manifold(
    Point<spacedim>(inter_distance / 2.0, 0, 0));

  surface_tria.set_manifold(0, left_ball_surface_manifold);
  surface_tria.set_manifold(1, right_ball_surface_manifold);

  std::map<typename Triangulation<dim, spacedim>::cell_iterator,
           typename Triangulation<dim + 1, spacedim>::face_iterator>
    map_from_surface_mesh_to_volume_mesh;

  map_from_surface_mesh_to_volume_mesh =
    GridGenerator::extract_boundary_mesh(tria, surface_tria);

  std::ofstream out("surface.msh");
  write_msh_correct(surface_tria, out);

  return 0;
}
