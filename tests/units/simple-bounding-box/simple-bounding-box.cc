/**
 * \file simple-bounding-box.cc
 * This file verifies the SimpleBoundingBox class.
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

// Grid input and output
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <simple_bounding_box.h>

#include <fstream>


using namespace dealii;

template <int dim>
void
print_bbox_info(const SimpleBoundingBox<dim> &bbox)
{
  deallog << "Volume:\t" << bbox.volume() << "\n";
  deallog << "Longest dimension: "
          << bbox.coordinate_index_with_longest_dimension() << "\n";
  const auto &boundary_points = bbox.get_boundary_points();
  deallog << "Bottom corner:\t" << boundary_points.first << "\n";
  deallog << "Top corner:\t" << boundary_points.second << std::endl;

  Point<dim> p1(1, 1, 1);
  Point<dim> p2(5, 9, 10);
  deallog << "Point (" << p1 << ") is "
          << (bbox.point_inside(p1) ? "inside" : "outside") << " the box\n";
  deallog << "Point (" << p2 << ") is "
          << (bbox.point_inside(p2) ? "inside" : "outside") << " the box"
          << std::endl;
}

int
main()
{
  /**
   * Initialize deal.ii log stream.
   */
  deallog.pop();
  deallog.depth_console(2);

  /**
   * Generate the grid for a 3D sphere.
   */
  const unsigned int      dim = 3;
  Triangulation<dim, dim> triangulation;
  GridGenerator::hyper_ball(triangulation, Point<3>(0., 0., 0.), 2.0, true);
  triangulation.refine_global(2);

  /**
   * Save the mesh to a file for visualization.
   */
  GridOut       grid_out;
  std::ofstream mesh_file("ball.msh");
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Generate a bounding box for the
   * triangulation using the vertex coordinates of the triangulation.
   */
  SimpleBoundingBox<dim, double> bbox(triangulation.get_vertices());
  std::cout << "Total number of vertices: " << triangulation.n_vertices()
            << "\n";

  /**
   * Print geometric information of the
   * bounding box.
   */
  print_bbox_info(bbox);

  /**
   * Divide the bounding box into halves.
   */
  std::pair<SimpleBoundingBox<dim, double>, SimpleBoundingBox<dim, double>>
    bbox_children = bbox.divide_geometrically();

  /**
   * Print geometric information of the two children boxes.
   */
  print_bbox_info(bbox_children.first);
  print_bbox_info(bbox_children.second);

  /**
   * Create a high order Lagrangian finite element.
   */
  const unsigned int fe_order = 2;
  FE_Q<dim, dim>     fe(fe_order);

  /**
   * Create a DoFHandler, which is associated with the triangulation and
   * distributed with the finite element.
   */
  DoFHandler<dim, dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  /**
   * Create a 2nd order mapping, which is required in generating the map from
   * DoF indices to support points.
   */
  const MappingQGeneric<dim, dim> mapping(fe_order);

  /**
   * Create bounding box based on the high order mapping.
   */
  SimpleBoundingBox<dim, double> bbox_dof(mapping, dof_handler);

  /**
   * Print geometric information of the
   * bounding box.
   */
  print_bbox_info(bbox_dof);

  /**
   * Create a DoF index set.
   */
  std::vector<types::global_dof_index> dof_index_set{1, 5, 10, 12};

  /**
   * Get the vector of support points.
   */
  std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  std::cout << "Total number of DoFs: " << dof_handler.n_dofs() << "\n";
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  /**
   * Create a bounding box for the DoF index set.
   */
  SimpleBoundingBox<dim, double> bbox_dof_index(dof_index_set, support_points);

  /**
   * Print geometric information of the
   * bounding box.
   */
  print_bbox_info(bbox_dof_index);

  dof_handler.clear();

  return 0;
}
