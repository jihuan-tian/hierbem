/**
 * @file grid_in_ext.h
 * @brief Introduction of grid_in_ext.h
 *
 * @date 2024-08-15
 * @author Jihuan Tian
 */
#ifndef INCLUDE_GRID_IN_EXT_H_
#define INCLUDE_GRID_IN_EXT_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <gmsh.h>

namespace HierBEM
{
  using namespace dealii;

  /**
   * Declaration of deal.ii internal functions which will be used in
   * @p read_skeleton_mesh.
   */
  template <int spacedim>
  void
  assign_1d_boundary_ids(
    const std::map<unsigned int, types::boundary_id> &boundary_ids,
    Triangulation<1, spacedim>                       &triangulation)
  {
    if (boundary_ids.size() > 0)
      for (const auto &cell : triangulation.active_cell_iterators())
        for (unsigned int f : GeometryInfo<1>::face_indices())
          if (boundary_ids.find(cell->vertex_index(f)) != boundary_ids.end())
            {
              AssertThrow(
                cell->at_boundary(f),
                ExcMessage(
                  "You are trying to prescribe boundary ids on the face "
                  "of a 1d cell (i.e., on a vertex), but this face is not actually at "
                  "the boundary of the mesh. This is not allowed."));
              cell->face(f)->set_boundary_id(
                boundary_ids.find(cell->vertex_index(f))->second);
            }
  }

  template <int dim, int spacedim>
  void
  assign_1d_boundary_ids(const std::map<unsigned int, types::boundary_id> &,
                         Triangulation<dim, spacedim> &)
  {
    // we shouldn't get here since boundary ids are not assigned to
    // vertices except in 1d
    Assert(dim != 1, ExcInternalError());
  }

  /**
   * Read mesh by assigning @p entity_tag as @p material_id for each cell.
   *
   * This @p entity_tag can be used to collect cells belonging to a surface
   * in DDM.
   *
   * @pre
   * @post
   * @tparam dim
   * @tparam spacedim
   * @param mesh_file File name for the mesh in MSH format.
   * @param tria Triangulation object.
   */
  template <int dim, int spacedim>
  void
  read_skeleton_mesh(const std::string            &mesh_file,
                     Triangulation<dim, spacedim> &tria)
  {
    // gmsh -> deal.II types
    const std::map<int, std::uint8_t> gmsh_to_dealii_type = {
      {{15, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {7, 5}, {6, 6}, {5, 7}}};

    // Vertex renumbering, by dealii type
    const std::array<std::vector<unsigned int>, 8> gmsh_to_dealii = {
      {{0},
       {{0, 1}},
       {{0, 1, 2}},
       {{0, 1, 3, 2}},
       {{0, 1, 2, 3}},
       {{0, 1, 3, 2, 4}},
       {{0, 1, 2, 3, 4, 5}},
       {{0, 1, 3, 2, 4, 5, 7, 6}}}};

    std::vector<Point<spacedim>>               vertices;
    std::vector<CellData<dim>>                 cells;
    SubCellData                                subcelldata;
    std::map<unsigned int, types::boundary_id> boundary_ids_1d;

    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", 0);
    gmsh::open(mesh_file);

    AssertThrow(gmsh::model::getDimension() == dim,
                ExcMessage(
                  "You are trying to read a gmsh file with dimension " +
                  std::to_string(gmsh::model::getDimension()) +
                  " into a grid of dimension " + std::to_string(dim)));

    // Read all nodes, and store them in our vector of vertices. Before we do
    // that, make sure all tags are consecutive
    {
      gmsh::model::mesh::removeDuplicateNodes();
      gmsh::model::mesh::renumberNodes();
      std::vector<std::size_t> node_tags;
      std::vector<double>      coord;
      std::vector<double>      parametricCoord;
      gmsh::model::mesh::getNodes(
        node_tags, coord, parametricCoord, -1, -1, false, false);
      vertices.resize(node_tags.size());
      for (unsigned int i = 0; i < node_tags.size(); ++i)
        {
          // Check that renumbering worked!
          AssertDimension(node_tags[i], i + 1);
          for (unsigned int d = 0; d < spacedim; ++d)
            vertices[i][d] = coord[i * 3 + d];
#ifdef DEBUG
          // Make sure the embedded dimension is right
          for (unsigned int d = spacedim; d < 3; ++d)
            Assert(std::abs(coord[i * 3 + d]) < 1e-10,
                   ExcMessage(
                     "The grid you are reading contains nodes that are "
                     "nonzero in the coordinate with index " +
                     std::to_string(d) +
                     ", but you are trying to save "
                     "it on a grid embedded in a " +
                     std::to_string(spacedim) + " dimensional space."));
#endif
        }
    }

    // TJH: Get all the elementary entities of dimension @p dim in the model, as a
    // vector of (dimension, tag) pairs:
    std::vector<std::pair<int, int>> entities;
    gmsh::model::getEntities(entities, dim);

    for (const auto &e : entities)
      {
        // Dimension and tag of the entity:
        const int &entity_dim = e.first;
        const int &entity_tag = e.second;

        types::manifold_id manifold_id = numbers::flat_manifold_id;
        types::boundary_id boundary_id = 0;

        // Get the physical tags, to deduce boundary, material, and
        // manifold_id
        std::vector<int> physical_tags;
        gmsh::model::getPhysicalGroupsForEntity(entity_dim,
                                                entity_tag,
                                                physical_tags);

        // Now fill manifold id and boundary or material id
        if (physical_tags.size())
          for (auto physical_tag : physical_tags)
            {
              std::string name;
              gmsh::model::getPhysicalName(entity_dim, physical_tag, name);
              if (!name.empty())
                try
                  {
                    std::map<std::string, int> id_names;
                    Patterns::Tools::to_value(name, id_names);
                    bool throw_anyway      = false;
                    bool found_boundary_id = false;
                    // If the above did not throw, we keep going, and retrieve
                    // all the information that we know how to translate.
                    for (const auto &it : id_names)
                      {
                        const auto &name = it.first;
                        const auto &id   = it.second;
                        if (entity_dim == dim && name == "MaterialID")
                          {
                            boundary_id = static_cast<types::boundary_id>(id);
                            found_boundary_id = true;
                          }
                        else if (entity_dim < dim && name == "BoundaryID")
                          {
                            boundary_id = static_cast<types::boundary_id>(id);
                            found_boundary_id = true;
                          }
                        else if (name == "ManifoldID")
                          manifold_id = static_cast<types::manifold_id>(id);
                        else
                          // We did not recognize one of the keys. We'll fall
                          // back to setting the boundary id to the physical
                          // tag after reading all strings.
                          throw_anyway = true;
                      }
                    // If we didn't find a BoundaryID:XX or MaterialID:XX, and
                    // something was found but not recognized, then we set the
                    // material id or boundary id in the catch block below,
                    // using directly the physical tag
                    if (throw_anyway && !found_boundary_id)
                      throw;
                  }
                catch (...)
                  {
                    // When the above didn't work, we revert to the old
                    // behaviour: the physical tag itself is interpreted
                    // either as a material_id or a boundary_id, and no
                    // manifold id is known
                    boundary_id = physical_tag;
                  }
            }

        // Get the mesh elements for the entity (dim, tag):
        std::vector<int>                      element_types;
        std::vector<std::vector<std::size_t>> element_ids, element_nodes;
        gmsh::model::mesh::getElements(
          element_types, element_ids, element_nodes, entity_dim, entity_tag);

        for (unsigned int i = 0; i < element_types.size(); ++i)
          {
            const auto &type       = gmsh_to_dealii_type.at(element_types[i]);
            const auto  n_vertices = gmsh_to_dealii[type].size();
            const auto &elements   = element_ids[i];
            const auto &nodes      = element_nodes[i];
            for (unsigned int j = 0; j < elements.size(); ++j)
              {
                if (entity_dim == dim)
                  {
                    cells.emplace_back(n_vertices);
                    auto &cell = cells.back();
                    for (unsigned int v = 0; v < n_vertices; ++v)
                      cell.vertices[v] =
                        nodes[n_vertices * j + gmsh_to_dealii[type][v]] - 1;
                    cell.manifold_id = manifold_id;
                    // TJH: Assign the @p entity_tag as @p material_id, which will be
                    // used to collect cells belong to a surface in DDM.
                    cell.material_id = entity_tag;
                  }
                else if (entity_dim == 2)
                  {
                    subcelldata.boundary_quads.emplace_back(n_vertices);
                    auto &face = subcelldata.boundary_quads.back();
                    for (unsigned int v = 0; v < n_vertices; ++v)
                      face.vertices[v] =
                        nodes[n_vertices * j + gmsh_to_dealii[type][v]] - 1;

                    face.manifold_id = manifold_id;
                    face.boundary_id = boundary_id;
                  }
                else if (entity_dim == 1)
                  {
                    subcelldata.boundary_lines.emplace_back(n_vertices);
                    auto &line = subcelldata.boundary_lines.back();
                    for (unsigned int v = 0; v < n_vertices; ++v)
                      line.vertices[v] =
                        nodes[n_vertices * j + gmsh_to_dealii[type][v]] - 1;

                    line.manifold_id = manifold_id;
                    line.boundary_id = boundary_id;
                  }
                else if (entity_dim == 0)
                  {
                    // TJH: This is only effective in one dimension.
                    if (dim == 1)
                      for (unsigned int j = 0; j < elements.size(); ++j)
                        boundary_ids_1d[nodes[j] - 1] = boundary_id;
                  }
              }
          }
      }

    Assert(subcelldata.check_consistency(dim), ExcInternalError());

    tria.create_triangulation_without_orientation_checking(vertices,
                                                           cells,
                                                           subcelldata);

    // in 1d, we also have to attach boundary ids to vertices, which does not
    // currently work through the call above
    if (dim == 1)
      assign_1d_boundary_ids(boundary_ids_1d, tria);

    gmsh::clear();
    gmsh::finalize();
  }
} // namespace HierBEM


#endif /* INCLUDE_GRID_IN_EXT_H_ */
