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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <gmsh.h>

namespace HierBEM
{
  using namespace dealii;

  /**
   * Declaration of deal.ii internal functions which will be used in
   * @p read_msh.
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
   * This function is a modification of dealii::read_msh(std::ifstream &).
   * 1. Some operatations in the original function can be enable/disabled by the
   * input arguments.
   * 2. For gmsh_file_format < 40, read the elementary tag as material id.
   *
   * @param read_lines_as_subcelldata When it is false, we do not read mesh data
   * in the following case: <code> (cell_type == 1) && ((dim == 2) || (dim ==
   * 3))</code>
   * This means lines in the mesh are not added to @p subcelldata.
   * @param reorder_cell_vertices When it is false, we do not execute
   * GridTools::consistently_order_cells(cells).
   * @param check_cell_orientation When it is false, we replace
   * @p tria.create_triangulation() at the end of this function with
   * @p tria.create_triangulation_without_orientation_checking. This disable
   * deal.ii's enforcing the manifold to be homeomorphic to a ball.
   */
  template <int dim, int spacedim>
  void
  read_msh(std::istream                 &in,
           Triangulation<dim, spacedim> &tria,
           const bool                    read_lines_as_subcelldata = false,
           const bool                    reorder_cell_vertices     = true,
           const bool                    check_cell_orientation    = false)
  {
    AssertThrow(in.fail() == false, ExcIO());

    unsigned int n_vertices;
    unsigned int n_cells;
    unsigned int dummy;
    std::string  line;
    // This array stores maps from the 'entities' to the 'physical tags' for
    // points, curves, surfaces and volumes. We use this information later to
    // assign boundary ids.
    std::array<std::map<int, int>, 4> tag_maps;

    in >> line;

    // first determine file format
    unsigned int gmsh_file_format = 0;
    if (line == "$NOD")
      gmsh_file_format = 10;
    else if (line == "$MeshFormat")
      gmsh_file_format = 20;
    else
      AssertThrow(false,
                  (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(line)));

    // if file format is 2.0 or greater then we also have to read the rest of
    // the header
    if (gmsh_file_format == 20)
      {
        double       version;
        unsigned int file_type, data_size;

        in >> version >> file_type >> data_size;

        Assert((version >= 2.0) && (version <= 4.1), ExcNotImplemented());
        gmsh_file_format = static_cast<unsigned int>(version * 10);

        Assert(file_type == 0, ExcNotImplemented());
        Assert(data_size == sizeof(double), ExcNotImplemented());

        // read the end of the header and the first line of the nodes
        // description to synch ourselves with the format 1 handling above
        in >> line;
        AssertThrow(line == "$EndMeshFormat",
                    (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(
                      line)));

        in >> line;
        // if the next block is of kind $PhysicalNames, ignore it
        if (line == "$PhysicalNames")
          {
            do
              {
                in >> line;
            } while (line != "$EndPhysicalNames");
            in >> line;
          }

        // if the next block is of kind $Entities, parse it
        if (line == "$Entities")
          {
            unsigned long n_points, n_curves, n_surfaces, n_volumes;

            in >> n_points >> n_curves >> n_surfaces >> n_volumes;
            for (unsigned int i = 0; i < n_points; ++i)
              {
                // parse point ids
                int          tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y,
                  box_max_z;

                // we only care for 'tag' as key for tag_maps[0]
                if (gmsh_file_format > 40)
                  {
                    in >> tag >> box_min_x >> box_min_y >> box_min_z >>
                      n_physicals;
                    box_max_x = box_min_x;
                    box_max_y = box_min_y;
                    box_max_z = box_min_z;
                  }
                else
                  {
                    in >> tag >> box_min_x >> box_min_y >> box_min_z >>
                      box_max_x >> box_max_y >> box_max_z >> n_physicals;
                  }
                // if there is a physical tag, we will use it as boundary id
                // below
                AssertThrow(n_physicals < 2,
                            ExcMessage("More than one tag is not supported!"));
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                  in >> physical_tag;
                tag_maps[0][tag] = physical_tag;
              }
            for (unsigned int i = 0; i < n_curves; ++i)
              {
                // parse curve ids
                int          tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y,
                  box_max_z;

                // we only care for 'tag' as key for tag_maps[1]
                in >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >>
                  box_max_y >> box_max_z >> n_physicals;
                // if there is a physical tag, we will use it as boundary id
                // below
                AssertThrow(n_physicals < 2,
                            ExcMessage("More than one tag is not supported!"));
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                  in >> physical_tag;
                tag_maps[1][tag] = physical_tag;
                // we don't care about the points associated to a curve, but
                // have to parse them anyway because their format is
                // unstructured
                in >> n_points;
                for (unsigned int j = 0; j < n_points; ++j)
                  in >> tag;
              }

            for (unsigned int i = 0; i < n_surfaces; ++i)
              {
                // parse surface ids
                int          tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y,
                  box_max_z;

                // we only care for 'tag' as key for tag_maps[2]
                in >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >>
                  box_max_y >> box_max_z >> n_physicals;
                // if there is a physical tag, we will use it as boundary id
                // below
                AssertThrow(n_physicals < 2,
                            ExcMessage("More than one tag is not supported!"));
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                  in >> physical_tag;
                tag_maps[2][tag] = physical_tag;
                // we don't care about the curves associated to a surface, but
                // have to parse them anyway because their format is
                // unstructured
                in >> n_curves;
                for (unsigned int j = 0; j < n_curves; ++j)
                  in >> tag;
              }
            for (unsigned int i = 0; i < n_volumes; ++i)
              {
                // parse volume ids
                int          tag;
                unsigned int n_physicals;
                double box_min_x, box_min_y, box_min_z, box_max_x, box_max_y,
                  box_max_z;

                // we only care for 'tag' as key for tag_maps[3]
                in >> tag >> box_min_x >> box_min_y >> box_min_z >> box_max_x >>
                  box_max_y >> box_max_z >> n_physicals;
                // if there is a physical tag, we will use it as boundary id
                // below
                AssertThrow(n_physicals < 2,
                            ExcMessage("More than one tag is not supported!"));
                // if there is no physical tag, use 0 as default
                int physical_tag = 0;
                for (unsigned int j = 0; j < n_physicals; ++j)
                  in >> physical_tag;
                tag_maps[3][tag] = physical_tag;
                // we don't care about the surfaces associated to a volume, but
                // have to parse them anyway because their format is
                // unstructured
                in >> n_surfaces;
                for (unsigned int j = 0; j < n_surfaces; ++j)
                  in >> tag;
              }
            in >> line;
            AssertThrow(line == "$EndEntities",
                        (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(
                          line)));
            in >> line;
          }

        // if the next block is of kind $PartitionedEntities, ignore it
        if (line == "$PartitionedEntities")
          {
            do
              {
                in >> line;
            } while (line != "$EndPartitionedEntities");
            in >> line;
          }

        // but the next thing should,
        // in any case, be the list of
        // nodes:
        AssertThrow(line == "$Nodes",
                    (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(
                      line)));
      }

    // now read the nodes list
    int n_entity_blocks = 1;
    if (gmsh_file_format > 40)
      {
        int min_node_tag;
        int max_node_tag;
        in >> n_entity_blocks >> n_vertices >> min_node_tag >> max_node_tag;
      }
    else if (gmsh_file_format == 40)
      {
        in >> n_entity_blocks >> n_vertices;
      }
    else
      in >> n_vertices;
    std::vector<Point<spacedim>> vertices(n_vertices);
    // set up mapping between numbering
    // in msh-file (nod) and in the
    // vertices vector
    std::map<int, int> vertex_indices;

    {
      unsigned int global_vertex = 0;
      for (int entity_block = 0; entity_block < n_entity_blocks; ++entity_block)
        {
          int           parametric;
          unsigned long numNodes;

          if (gmsh_file_format < 40)
            {
              numNodes   = n_vertices;
              parametric = 0;
            }
          else
            {
              // for gmsh_file_format 4.1 the order of tag and dim is reversed,
              // but we are ignoring both anyway.
              int tagEntity, dimEntity;
              in >> tagEntity >> dimEntity >> parametric >> numNodes;
            }

          std::vector<int> vertex_numbers;
          int              vertex_number;
          if (gmsh_file_format > 40)
            for (unsigned long vertex_per_entity = 0;
                 vertex_per_entity < numNodes;
                 ++vertex_per_entity)
              {
                in >> vertex_number;
                vertex_numbers.push_back(vertex_number);
              }

          for (unsigned long vertex_per_entity = 0;
               vertex_per_entity < numNodes;
               ++vertex_per_entity, ++global_vertex)
            {
              int    vertex_number;
              double x[3];

              // read vertex
              if (gmsh_file_format > 40)
                {
                  vertex_number = vertex_numbers[vertex_per_entity];
                  in >> x[0] >> x[1] >> x[2];
                }
              else
                in >> vertex_number >> x[0] >> x[1] >> x[2];

              for (unsigned int d = 0; d < spacedim; ++d)
                vertices[global_vertex](d) = x[d];
              // store mapping
              vertex_indices[vertex_number] = global_vertex;

              // ignore parametric coordinates
              if (parametric != 0)
                {
                  double u = 0.;
                  double v = 0.;
                  in >> u >> v;
                  (void)u;
                  (void)v;
                }
            }
        }
      AssertDimension(global_vertex, n_vertices);
    }

    // Assert we reached the end of the block
    in >> line;
    static const std::string end_nodes_marker[] = {"$ENDNOD", "$EndNodes"};
    AssertThrow(line == end_nodes_marker[gmsh_file_format == 10 ? 0 : 1],
                (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(line)));

    // Now read in next bit
    in >> line;
    static const std::string begin_elements_marker[] = {"$ELM", "$Elements"};
    AssertThrow(line == begin_elements_marker[gmsh_file_format == 10 ? 0 : 1],
                (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(line)));

    // now read the cell list
    if (gmsh_file_format > 40)
      {
        int min_node_tag;
        int max_node_tag;
        in >> n_entity_blocks >> n_cells >> min_node_tag >> max_node_tag;
      }
    else if (gmsh_file_format == 40)
      {
        in >> n_entity_blocks >> n_cells;
      }
    else
      {
        n_entity_blocks = 1;
        in >> n_cells;
      }

    // set up array of cells and subcells (faces). In 1d, there is currently no
    // standard way in deal.II to pass boundary indicators attached to
    // individual vertices, so do this by hand via the boundary_ids_1d array
    std::vector<CellData<dim>>                 cells;
    SubCellData                                subcelldata;
    std::map<unsigned int, types::boundary_id> boundary_ids_1d;
    bool                                       is_quad_or_hex_mesh = false;
    bool                                       is_tria_or_tet_mesh = false;

    {
      unsigned int global_cell = 0;
      for (int entity_block = 0; entity_block < n_entity_blocks; ++entity_block)
        {
          unsigned int  material_id;
          unsigned long numElements;
          int           cell_type;

          if (gmsh_file_format < 40)
            {
              material_id = 0;
              cell_type   = 0;
              numElements = n_cells;
            }
          else if (gmsh_file_format == 40)
            {
              int tagEntity, dimEntity;
              in >> tagEntity >> dimEntity >> cell_type >> numElements;
              material_id = tag_maps[dimEntity][tagEntity];
            }
          else
            {
              // for gmsh_file_format 4.1 the order of tag and dim is reversed,
              int tagEntity, dimEntity;
              in >> dimEntity >> tagEntity >> cell_type >> numElements;
              material_id = tag_maps[dimEntity][tagEntity];
            }

          for (unsigned int cell_per_entity = 0; cell_per_entity < numElements;
               ++cell_per_entity, ++global_cell)
            {
              // note that since in the input
              // file we found the number of
              // cells at the top, there
              // should still be input here,
              // so check this:
              AssertThrow(in.fail() == false, ExcIO());

              unsigned int nod_num;

              /*
                For file format version 1, the format of each cell is as
                follows: elm-number elm-type reg-phys reg-elem number-of-nodes
                node-number-list

                However, for version 2, the format reads like this:
                elm-number elm-type number-of-tags < tag > ...
                node-number-list

                For version 4, we have:
                tag(int) numVert(int) ...

                In the following, we will ignore the element number (we simply
                enumerate them in the order in which we read them, and we will
                take reg-phys (version 1) or the first tag (version 2, if any
                tag is given at all) as material id. For version 4, we already
                read the material and the cell type in above.
              */

              unsigned int elm_number = 0;
              if (gmsh_file_format < 40)
                {
                  in >> elm_number // ELM-NUMBER
                    >> cell_type;  // ELM-TYPE
                }

              if (gmsh_file_format < 20)
                {
                  in >> material_id // REG-PHYS
                    >> dummy        // reg_elm
                    >> nod_num;
                }
              else if (gmsh_file_format < 40)
                {
                  unsigned int n_tags;
                  in >> n_tags;
                  if (n_tags >= 2)
                    {
                      // TJH: Here we read the second tag, i.e. the elementary
                      // entity tag as the material_id.
                      in >> dummy;
                      in >> material_id;

                      // Consume the remaining tags.
                      for (unsigned int i = 2; i < n_tags; ++i)
                        in >> dummy;
                    }
                  else if (n_tags == 1)
                    {
                      // When there is only one tag, we still use the only tag
                      // as the material_id.
                      in >> material_id;
                    }
                  else
                    material_id = 0;

                  if (cell_type == 1) // line
                    nod_num = 2;
                  else if (cell_type == 2) // tri
                    nod_num = 3;
                  else if (cell_type == 3) // quad
                    nod_num = 4;
                  else if (cell_type == 4) // tet
                    nod_num = 4;
                  else if (cell_type == 5) // hex
                    nod_num = 8;
                }
              else // file format version 4.0 and later
                {
                  // ignore tag
                  int tag;
                  in >> tag;

                  if (cell_type == 1) // line
                    nod_num = 2;
                  else if (cell_type == 2) // tri
                    nod_num = 3;
                  else if (cell_type == 3) // quad
                    nod_num = 4;
                  else if (cell_type == 4) // tet
                    nod_num = 4;
                  else if (cell_type == 5) // hex
                    nod_num = 8;
                }


              /*       `ELM-TYPE'
                       defines the geometrical type of the N-th element:
                       `1'
                       Line (2 nodes, 1 edge).

                       `2'
                       Triangle (3 nodes, 3 edges).

                       `3'
                       Quadrangle (4 nodes, 4 edges).

                       `4'
                       Tetrahedron (4 nodes, 6 edges, 6 faces).

                       `5'
                       Hexahedron (8 nodes, 12 edges, 6 faces).

                       `15'
                       Point (1 node).
              */

              if (((cell_type == 1) && (dim == 1)) || // a line in 1d
                  ((cell_type == 2) && (dim == 2)) || // a triangle in 2d
                  ((cell_type == 3) && (dim == 2)) || // a quadrilateral in 2d
                  ((cell_type == 4) && (dim == 3)) || // a tet in 3d
                  ((cell_type == 5) && (dim == 3)))   // a hex in 3d
                // found a cell
                {
                  unsigned int vertices_per_cell = 0;
                  if (cell_type == 1) // line
                    vertices_per_cell = 2;
                  else if (cell_type == 2) // tri
                    {
                      vertices_per_cell   = 3;
                      is_tria_or_tet_mesh = true;
                    }
                  else if (cell_type == 3) // quad
                    {
                      vertices_per_cell   = 4;
                      is_quad_or_hex_mesh = true;
                    }
                  else if (cell_type == 4) // tet
                    {
                      vertices_per_cell   = 4;
                      is_tria_or_tet_mesh = true;
                    }
                  else if (cell_type == 5) // hex
                    {
                      vertices_per_cell   = 8;
                      is_quad_or_hex_mesh = true;
                    }

                  AssertThrow(nod_num == vertices_per_cell,
                              ExcMessage(
                                "Number of nodes does not coincide with the "
                                "number required for this object"));

                  // allocate and read indices
                  cells.emplace_back();
                  cells.back().vertices.resize(vertices_per_cell);
                  for (unsigned int i = 0; i < vertices_per_cell; ++i)
                    {
                      // hypercube cells need to be reordered
                      if (vertices_per_cell ==
                          GeometryInfo<dim>::vertices_per_cell)
                        {
                          in >> cells.back()
                                  .vertices[GeometryInfo<dim>::ucd_to_deal[i]];
                        }
                      else
                        {
                          in >> cells.back().vertices[i];
                        }
                    }

                  // to make sure that the cast won't fail
                  Assert(material_id <=
                           std::numeric_limits<types::material_id>::max(),
                         ExcIndexRange(
                           material_id,
                           0,
                           std::numeric_limits<types::material_id>::max()));
                  // we use only material_ids in the range from 0 to
                  // numbers::invalid_material_id-1
                  AssertIndexRange(material_id, numbers::invalid_material_id);

                  cells.back().material_id = material_id;

                  // transform from gmsh to consecutive numbering
                  for (unsigned int i = 0; i < vertices_per_cell; ++i)
                    {
                      AssertThrow(vertex_indices.find(
                                    cells.back().vertices[i]) !=
                                    vertex_indices.end(),
                                  (typename GridIn<dim, spacedim>::
                                     ExcInvalidVertexIndexGmsh(
                                       cell_per_entity,
                                       elm_number,
                                       cells.back().vertices[i])));

                      // vertex with this index exists
                      cells.back().vertices[i] =
                        vertex_indices[cells.back().vertices[i]];
                    }
                }
              else if ((cell_type == 1) &&
                       ((dim == 2) || (dim == 3))) // a line in 2d or 3d
                // boundary info
                {
                  if (read_lines_as_subcelldata)
                    {
                      subcelldata.boundary_lines.emplace_back();
                      in >> subcelldata.boundary_lines.back().vertices[0] >>
                        subcelldata.boundary_lines.back().vertices[1];


                      // to make sure that the cast won't fail
                      Assert(material_id <=
                               std::numeric_limits<types::boundary_id>::max(),
                             ExcIndexRange(
                               material_id,
                               0,
                               std::numeric_limits<types::boundary_id>::max()));
                      // we use only boundary_ids in the range from 0 to
                      // numbers::internal_face_boundary_id-1
                      AssertIndexRange(material_id,
                                       numbers::internal_face_boundary_id);

                      subcelldata.boundary_lines.back().boundary_id =
                        static_cast<types::boundary_id>(material_id);

                      // transform from ucd to
                      // consecutive numbering
                      for (unsigned int &vertex :
                           subcelldata.boundary_lines.back().vertices)
                        if (vertex_indices.find(vertex) != vertex_indices.end())
                          // vertex with this index exists
                          vertex = vertex_indices[vertex];
                        else
                          {
                            // no such vertex index
                            AssertThrow(false,
                                        (typename GridIn<dim, spacedim>::
                                           ExcInvalidVertexIndex(
                                             cell_per_entity, vertex)));
                            vertex = numbers::invalid_unsigned_int;
                          }
                    }
                  else
                    {
                      // TJH: Here we only read the data file but do not add any
                      // lines to the @p subcelldata, since in the current BEM
                      // solver, the boundary of the boundary of a volume is
                      // empty.
                      unsigned int v1, v2;
                      in >> v1 >> v2;
                    }
                }
              else if ((cell_type == 2 || cell_type == 3) &&
                       (dim == 3)) // a triangle or a quad in 3d
                // boundary info
                {
                  unsigned int vertices_per_cell = 0;
                  // check cell type
                  if (cell_type == 2) // tri
                    {
                      vertices_per_cell   = 3;
                      is_tria_or_tet_mesh = true;
                    }
                  else if (cell_type == 3) // quad
                    {
                      vertices_per_cell   = 4;
                      is_quad_or_hex_mesh = true;
                    }

                  subcelldata.boundary_quads.emplace_back();

                  // resize vertices
                  subcelldata.boundary_quads.back().vertices.resize(
                    vertices_per_cell);
                  // for loop
                  for (unsigned int i = 0; i < vertices_per_cell; ++i)
                    in >> subcelldata.boundary_quads.back().vertices[i];

                  // to make sure that the cast won't fail
                  Assert(material_id <=
                           std::numeric_limits<types::boundary_id>::max(),
                         ExcIndexRange(
                           material_id,
                           0,
                           std::numeric_limits<types::boundary_id>::max()));
                  // we use only boundary_ids in the range from 0 to
                  // numbers::internal_face_boundary_id-1
                  AssertIndexRange(material_id,
                                   numbers::internal_face_boundary_id);

                  subcelldata.boundary_quads.back().boundary_id =
                    static_cast<types::boundary_id>(material_id);

                  // transform from gmsh to
                  // consecutive numbering
                  for (unsigned int &vertex :
                       subcelldata.boundary_quads.back().vertices)
                    if (vertex_indices.find(vertex) != vertex_indices.end())
                      // vertex with this index exists
                      vertex = vertex_indices[vertex];
                    else
                      {
                        // no such vertex index
                        Assert(false,
                               (typename GridIn<dim, spacedim>::
                                  ExcInvalidVertexIndex(cell_per_entity,
                                                        vertex)));
                        vertex = numbers::invalid_unsigned_int;
                      }
                }
              else if (cell_type == 15)
                {
                  // read the indices of nodes given
                  unsigned int node_index = 0;
                  if (gmsh_file_format < 20)
                    {
                      // For points (cell_type==15), we can only ever
                      // list one node index.
                      AssertThrow(nod_num == 1, ExcInternalError());
                      in >> node_index;
                    }
                  else
                    {
                      in >> node_index;
                    }

                  // we only care about boundary indicators assigned to
                  // individual vertices in 1d (because otherwise the vertices
                  // are not faces)
                  if (dim == 1)
                    boundary_ids_1d[vertex_indices[node_index]] = material_id;
                }
              else
                {
                  AssertThrow(
                    false,
                    (typename GridIn<dim, spacedim>::ExcGmshUnsupportedGeometry(
                      cell_type)));
                }
            }
        }
      AssertDimension(global_cell, n_cells);
    }
    // Assert that we reached the end of the block
    in >> line;
    static const std::string end_elements_marker[] = {"$ENDELM",
                                                      "$EndElements"};
    AssertThrow(line == end_elements_marker[gmsh_file_format == 10 ? 0 : 1],
                (typename GridIn<dim, spacedim>::ExcInvalidGMSHInput(line)));

    // check that no forbidden arrays are used
    Assert(subcelldata.check_consistency(dim), ExcInternalError());

    AssertThrow(in.fail() == false, ExcIO());

    // check that we actually read some cells.
    AssertThrow(cells.size() > 0,
                (typename GridIn<dim, spacedim>::ExcGmshNoCellInformation(
                  subcelldata.boundary_lines.size(),
                  subcelldata.boundary_quads.size())));

    // TODO: the functions below (GridTools::delete_unused_vertices(),
    // GridTools::invert_all_negative_measure_cells(),
    // GridTools::consistently_order_cells()) need to be revisited
    // for simplex/mixed meshes

    if (dim == 1 || (is_quad_or_hex_mesh && !is_tria_or_tet_mesh))
      {
        // do some clean-up on vertices...
        GridTools::delete_unused_vertices(vertices, cells, subcelldata);
        // ... and cells
        if (dim == spacedim)
          GridTools::invert_cells_with_negative_measure(vertices, cells);
        if (reorder_cell_vertices)
          GridTools::consistently_order_cells(cells);
      }
    else if (is_tria_or_tet_mesh)
      {
        if (dim == spacedim)
          GridTools::invert_cells_with_negative_measure(vertices, cells);
      }

    if (check_cell_orientation)
      tria.create_triangulation(vertices, cells, subcelldata);
    else
      tria.create_triangulation_without_orientation_checking(vertices,
                                                             cells,
                                                             subcelldata);

    // in 1d, we also have to attach boundary ids to vertices, which does not
    // currently work through the call above
    if (dim == 1)
      assign_1d_boundary_ids(boundary_ids_1d, tria);
  }


  /**
   * This function is a modification of dealii::read_msh(string &).
   * 1. Some operatations in the original function can be enable/disabled by the
   * input arguments.
   * 2. Read mesh by assigning @p entity_tag as @p material_id for each cell. This
   * @p entity_tag can be used to collect cells belonging to a surface in DDM.
   * 3. Only read elementary entities with dimension @p dim.
   * 4. Disable the assertion about @p dim==1 when entity_dim == 0.
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
  read_msh(const std::string            &mesh_file,
           Triangulation<dim, spacedim> &tria,
           const bool                    check_cell_orientation = true)
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
    // vector of (dimension, tag) pairs. This is different from
    // dealii::read_msh(const std::string &fname), which read entities of all
    // dimensions.
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
                    // TJH: This is only effective in one dimension. In the
                    // original function dealii::read_msh(const std::string
                    // &fname), here is an assertion about @p dim==1. Since a
                    // mesh generated from a CAD model (when there are no
                    // physical groups defined, "save all" should be enabled in
                    // Gmsh to save the mesh, otherwise, nothing will be saved)
                    // contains geometric entities of all dimensions, such as a
                    // vertex with zero dimension, it will make the original
                    // assertion fail.
                    if (dim == 1)
                      for (unsigned int j = 0; j < elements.size(); ++j)
                        boundary_ids_1d[nodes[j] - 1] = boundary_id;
                  }
              }
          }
      }

    Assert(subcelldata.check_consistency(dim), ExcInternalError());

    if (check_cell_orientation)
      tria.create_triangulation(vertices, cells, subcelldata);
    else
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
