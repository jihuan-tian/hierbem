/**
 * @file grid_out_ext.h
 * @brief Introduction of grid_out_ext.h
 *
 * @date 2022-11-16
 * @author Jihuan Tian
 */
#ifndef INCLUDE_GRID_OUT_EXT_H_
#define INCLUDE_GRID_OUT_EXT_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>

using namespace dealii;

template <int dim, int spacedim>
void
write_msh_correct(const Triangulation<dim, spacedim> &tria, std::ostream &out)
{
  AssertThrow(out, ExcIO());

  // get the positions of the
  // vertices and whether they are
  // used.
  const std::vector<Point<spacedim>> &vertices    = tria.get_vertices();
  const std::vector<bool> &           vertex_used = tria.get_used_vertices();

  const unsigned int n_vertices = tria.n_used_vertices();

  typename Triangulation<dim, spacedim>::active_cell_iterator cell =
    tria.begin_active();
  const typename Triangulation<dim, spacedim>::active_cell_iterator endc =
    tria.end();

  // Write Header
  // The file format is:
  /*
  $NOD
  number-of-nodes
  node-number x-coord y-coord z-coord
  ...
  $ENDNOD
  $ELM
  number-of-elements
  elm-number elm-type number-of-tags reg-phys reg-elem node-number-list
  ...
  $ENDELM
  */
  out << "$MeshFormat\n"
      << "2.2 0 8\n"
      << "$EndMeshFormat\n"
      << "$Nodes" << '\n'
      << n_vertices << '\n';

  // actually write the vertices.
  // note that we shall number them
  // with first index 1 instead of 0
  for (unsigned int i = 0; i < vertices.size(); ++i)
    if (vertex_used[i])
      {
        out << i + 1 // vertex index
            << "  " << vertices[i];
        for (unsigned int d = spacedim + 1; d <= 3; ++d)
          out << " 0"; // fill with zeroes
        out << '\n';
      }

  // Write cells preamble
  out << "$EndNodes" << '\n'
      << "$Elements" << '\n'
      << tria.n_active_cells() << '\n';

  /*
    elm-type
    defines the geometrical type of the n-th element:
    1
    Line (2 nodes).
    2
    Triangle (3 nodes).
    3
    Quadrangle (4 nodes).
    4
    Tetrahedron (4 nodes).
    5
    Hexahedron (8 nodes).
    6
    Prism (6 nodes).
    7
    Pyramid (5 nodes).
    8
    Second order line (3 nodes: 2 associated with the vertices and 1 with the
    edge).
    9
    Second order triangle (6 nodes: 3 associated with the vertices and 3 with
    the edges). 10 Second order quadrangle (9 nodes: 4 associated with the
    vertices, 4 with the edges and 1 with the face). 11 Second order tetrahedron
    (10 nodes: 4 associated with the vertices and 6 with the edges). 12 Second
    order hexahedron (27 nodes: 8 associated with the vertices, 12 with the
    edges, 6 with the faces and 1 with the volume). 13 Second order prism (18
    nodes: 6 associated with the vertices, 9 with the edges and 3 with the
    quadrangular faces). 14 Second order pyramid (14 nodes: 5 associated with
    the vertices, 8 with the edges and 1 with the quadrangular face). 15 Point
    (1 node).
  */
  unsigned int elm_type;
  switch (dim)
    {
      case 1:
        elm_type = 1;
        break;
      case 2:
        elm_type = 3;
        break;
      case 3:
        elm_type = 5;
        break;
      default:
        Assert(false, ExcNotImplemented());
    }

  // write cells. Enumerate cells
  // consecutively, starting with 1
  const unsigned int number_of_tags = 2;
  for (cell = tria.begin_active(); cell != endc; ++cell)
    {
      out << cell->active_cell_index() + 1 << ' ' << elm_type << ' '
          << number_of_tags << ' '
          << static_cast<unsigned int>(cell->material_id()) << ' '
          << static_cast<unsigned int>(cell->material_id()) << ' ';

      // Vertex numbering follows UCD conventions.

      for (unsigned int vertex = 0;
           vertex < GeometryInfo<dim>::vertices_per_cell;
           ++vertex)
        out << cell->vertex_index(GeometryInfo<dim>::ucd_to_deal[vertex]) + 1
            << ' ';
      out << '\n';
    }

  out << "$EndElements\n";

  // make sure everything now gets to
  // disk
  out.flush();

  AssertThrow(out, ExcIO());
}

#endif /* INCLUDE_GRID_OUT_EXT_H_ */
