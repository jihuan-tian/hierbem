/**
 * @file ddm_efield.h
 * @brief Introduction of ddm_efield.h
 *
 * @date 2024-07-26
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_H_
#define INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>

#include <gmsh.h>

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "aca_plus.hcu"
#include "config.h"
#include "ddm_efield_global_preconditioner.h"
#include "ddm_efield_matrix.h"
#include "dof_tools_ext.h"
#include "gmsh_manipulation.h"
#include "grid_out_ext.h"
#include "mapping/mapping_info.h"
#include "subdomain_steklov_poincare_hmatrix.h"

namespace HierBEM
{
  using namespace dealii;

  enum SubdomainType
  {
    SurroundingSpace,
    Dielectric,
    VoltageConductor,
    FloatingConductor
  };


  template <int dim, int spacedim>
  class SubdomainTopology
  {
  public:
    SubdomainTopology() = default;

    void
    generate_topology(const std::string &cad_file,
                      const std::string &mesh_file,
                      const double       eps_for_orientation_detection = 1e-5);

    void
    print(std::ostream &out) const;

    const std::map<EntityTag, std::array<EntityTag, 2>> &
    get_surface_to_subdomain() const
    {
      return surface_to_subdomain;
    }

    const std::map<EntityTag, std::vector<EntityTag>> &
    get_subdomain_to_surface() const
    {
      return subdomain_to_surface;
    }

    std::map<EntityTag, std::array<EntityTag, 2>> &
    get_surface_to_subdomain()
    {
      return surface_to_subdomain;
    }

    std::map<EntityTag, std::vector<EntityTag>> &
    get_subdomain_to_surface()
    {
      return subdomain_to_surface;
    }

  private:
    std::map<EntityTag, std::vector<EntityTag>>   subdomain_to_surface;
    std::map<EntityTag, std::array<EntityTag, 2>> surface_to_subdomain;
  };


  template <int dim, int spacedim>
  void
  SubdomainTopology<dim, spacedim>::generate_topology(
    const std::string &cad_file,
    const std::string &mesh_file,
    const double       eps_for_orientation_detection)
  {
    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", 0);
    gmsh::open(cad_file);
    gmsh::merge(mesh_file);
    gmsh::model::occ::synchronize();

    Assert(gmsh::model::getDimension() >= dim + 1, ExcInternalError());

    // Get all volume entities by first trying the OCC kernel. If the number of
    // returned entities is zero, try Gmsh's own CAD kernel.
    gmsh::vectorpair volume_dimtag_list;
    gmsh::model::occ::getEntities(volume_dimtag_list, dim + 1);
    if (volume_dimtag_list.size() == 0)
      {
        gmsh::model::getEntities(volume_dimtag_list, dim + 1);
      }

    Assert(volume_dimtag_list.size() > 0, ExcInternalError());

    // The boundary entities of each volume entity.
    std::vector<EntityTag> oriented_surface_tags;
    for (const auto &volume_dimtag : volume_dimtag_list)
      {
        GmshManip<dim, spacedim>::get_oriented_volume_boundaries(
          volume_dimtag.second,
          oriented_surface_tags,
          surface_to_subdomain,
          eps_for_orientation_detection);

        subdomain_to_surface[volume_dimtag.second] = oriented_surface_tags;
      }

    gmsh::clear();
    gmsh::finalize();
  }


  template <int dim, int spacedim>
  void
  SubdomainTopology<dim, spacedim>::print(std::ostream &out) const
  {
    out << "=== subdomain-to-face ===\n";

    for (const auto &record : subdomain_to_surface)
      {
        out << record.first << ":";
        for (const auto face_tag : record.second)
          {
            out << " " << face_tag;
          }
        out << "\n";
      }

    out << "=== face-to-subdomain ===\n";
    for (const auto &record : surface_to_subdomain)
      {
        out << record.first << ": " << record.second[0] << " "
            << record.second[1] << "\n";
      }

    out << std::endl;
  }

  template <int spacedim>
  class EfieldSubdomain;

  template <int spacedim>
  class EfieldSurface
  {
  public:
    EfieldSurface(const EntityTag             entity_tag,
                  EfieldSurface<spacedim>    *neighbor_surface,
                  EfieldSubdomain<spacedim>  *parent_subdomain,
                  const bool                  is_normal_outward,
                  const bool                  is_dirichlet_boundary,
                  Function<spacedim, double> *dirichlet_voltage);

    EfieldSurface(const EfieldSurface &surface) = default;

    EfieldSurface &
    operator=(const EfieldSurface &surface) = default;

    Function<spacedim, double> *
    get_dirichlet_voltage()
    {
      return dirichlet_voltage;
    }

    void
    set_dirichlet_voltage(Function<spacedim, double> *dirichletVoltage)
    {
      dirichlet_voltage = dirichletVoltage;
    }

    EntityTag
    get_entity_tag() const
    {
      return entity_tag;
    }

    void
    set_entity_tag(const EntityTag entityTag)
    {
      entity_tag = entityTag;
    }

    bool
    get_is_dirichlet_boundary() const
    {
      return is_dirichlet_boundary;
    }

    void
    set_is_dirichlet_boundary(const bool isDirichletBoundary)
    {
      is_dirichlet_boundary = isDirichletBoundary;
    }

    bool
    get_is_normal_outward() const
    {
      return is_normal_outward;
    }

    void
    set_is_normal_outward(const bool isNormalOutward)
    {
      is_normal_outward = isNormalOutward;
    }

    EfieldSurface<spacedim> *
    get_neighbor_surface()
    {
      return neighbor_surface;
    }

    void
    set_neighbor_surface(EfieldSurface<spacedim> *neighborSurface)
    {
      neighbor_surface = neighborSurface;
    }

    EfieldSubdomain<spacedim> *
    get_parent_subdomain()
    {
      return parent_subdomain;
    }

    void
    set_parent_subdomain(EfieldSubdomain<spacedim> *parentSubdomain)
    {
      parent_subdomain = parentSubdomain;
    }

  private:
    EntityTag                   entity_tag;
    EfieldSurface<spacedim>    *neighbor_surface;
    EfieldSubdomain<spacedim>  *parent_subdomain;
    bool                        is_normal_outward;
    bool                        is_dirichlet_boundary;
    Function<spacedim, double> *dirichlet_voltage;
  };


  template <int spacedim>
  EfieldSurface<spacedim>::EfieldSurface(
    const EntityTag             entity_tag,
    EfieldSurface<spacedim>    *neighbor_surface,
    EfieldSubdomain<spacedim>  *parent_subdomain,
    const bool                  is_normal_outward,
    const bool                  is_dirichlet_boundary,
    Function<spacedim, double> *dirichlet_voltage)
    : entity_tag(entity_tag)
    , neighbor_surface(neighbor_surface)
    , parent_subdomain(parent_subdomain)
    , is_normal_outward(is_normal_outward)
    , is_dirichlet_boundary(is_dirichlet_boundary)
    , dirichlet_voltage(dirichlet_voltage)
  {}


  /**
   * A subdomain in electric field problem
   */
  template <int spacedim>
  class EfieldSubdomain
  {
  public:
    EfieldSubdomain() = default;

    EfieldSubdomain(const EntityTag     entity_tag,
                    const SubdomainType type,
                    const double        permittivity,
                    const double        voltage);

    EfieldSubdomain(const EfieldSubdomain &subdomain) = default;

    EfieldSubdomain &
    operator=(const EfieldSubdomain &subdomain) = default;

    const std::vector<EfieldSurface<spacedim>> &
    get_surfaces_touching_dielectric() const
    {
      return surfaces_touching_dielectric;
    }

    const std::vector<EfieldSurface<spacedim>> &
    get_surfaces_touching_floating_conductor() const
    {
      return surfaces_touching_floating_conductor;
    }

    const std::vector<EfieldSurface<spacedim>> &
    get_surfaces_touching_voltage_conductor() const
    {
      return surfaces_touching_voltage_conductor;
    }

    std::vector<EfieldSurface<spacedim>> &
    get_surfaces_touching_dielectric()
    {
      return surfaces_touching_dielectric;
    }

    std::vector<EfieldSurface<spacedim>> &
    get_surfaces_touching_floating_conductor()
    {
      return surfaces_touching_floating_conductor;
    }

    std::vector<EfieldSurface<spacedim>> &
    get_surfaces_touching_voltage_conductor()
    {
      return surfaces_touching_voltage_conductor;
    }

    EntityTag
    get_entity_tag() const
    {
      return entity_tag;
    }

    double
    get_permittivity() const
    {
      return permittivity;
    }

    SubdomainType
    get_type() const
    {
      return type;
    }

    double
    get_voltage() const
    {
      return voltage;
    }

  private:
    EntityTag                            entity_tag;
    SubdomainType                        type;
    double                               permittivity;
    double                               voltage;
    std::vector<EfieldSurface<spacedim>> surfaces_touching_dielectric;
    std::vector<EfieldSurface<spacedim>> surfaces_touching_voltage_conductor;
    std::vector<EfieldSurface<spacedim>> surfaces_touching_floating_conductor;
  };


  template <int spacedim>
  EfieldSubdomain<spacedim>::EfieldSubdomain(const EntityTag     entity_tag,
                                             const SubdomainType type,
                                             const double        permittivity,
                                             const double        voltage)
    : entity_tag(entity_tag)
    , type(type)
    , permittivity(permittivity)
    , voltage(voltage)
  {}


  template <int spacedim>
  class EfieldSubdomainDescription
  {
  public:
    EfieldSubdomainDescription() = default;

    const std::map<EntityTag, EfieldSubdomain<spacedim>> &
    get_subdomains() const
    {
      return subdomains;
    }

    const std::vector<EfieldSubdomain<spacedim> *> &
    get_dielectric_subdomains() const
    {
      return dielectric_subdomains;
    }

    const std::vector<EfieldSubdomain<spacedim> *> &
    get_floating_conductor_subdomains() const
    {
      return floating_conductor_subdomains;
    }

    const std::vector<EfieldSubdomain<spacedim> *> &
    get_voltage_conductor_subdomains() const
    {
      return voltage_conductor_subdomains;
    }

    std::vector<EfieldSubdomain<spacedim> *> &
    get_dielectric_subdomains()
    {
      return dielectric_subdomains;
    }

    std::vector<EfieldSubdomain<spacedim> *> &
    get_floating_conductor_subdomains()
    {
      return floating_conductor_subdomains;
    }

    std::vector<EfieldSubdomain<spacedim> *> &
    get_voltage_conductor_subdomains()
    {
      return voltage_conductor_subdomains;
    }

    std::map<EntityTag, EfieldSubdomain<spacedim>> &
    get_subdomains()
    {
      return subdomains;
    }

  private:
    std::map<EntityTag, EfieldSubdomain<spacedim>> subdomains;
    std::vector<EfieldSubdomain<spacedim> *>       dielectric_subdomains;
    std::vector<EfieldSubdomain<spacedim> *>       voltage_conductor_subdomains;
    std::vector<EfieldSubdomain<spacedim> *> floating_conductor_subdomains;
  };

  template <int dim, int spacedim>
  class DDMEfield
  {
  public:
    DDMEfield();

    ~DDMEfield();

    /**
     * Read mesh by assigning @p entity_tag as @p material_id for each cell.
     * This @p entity_tag will be used to collect cells belonging to a surface
     * in DDM.
     *
     * @pre
     * @post
     * @param mesh_file File name for the mesh in MSH format.
     */
    void
    read_skeleton_mesh(const std::string &mesh_file);

    /**
     * Read the CAD file and build the association relationship between volumes
     * and surfaces.
     *
     * @pre
     * @post
     * @param cad_file
     */
    void
    read_subdomain_topology(const std::string &cad_file,
                            const std::string &mesh_file);

    /**
     * Manually initialize problem parameters for testing purpose.
     *
     * @pre
     * @post
     */
    void
    initialize_parameters();

    void
    initialize_manifolds_and_mappings();

    /**
     * Interpolate Dirichlet boundary conditions on dielectric surfaces or
     * interfaces.
     *
     * @pre
     * @post
     */
    void
    interpolate_surface_dirichlet_bc();

    void
    create_efield_subdomains_and_surfaces();

    /**
     * Initialize \hmats for each dielectric subdomain.
     * @pre
     * @post
     */
    void
    initialize_subdomain_hmatrices();

    void
    setup_system();

    void
    assemble_system();

    void
    solve();

    void
    output_results();

    const SubdomainTopology<dim, spacedim> &
    get_subdomain_topology() const
    {
      return subdomain_topology;
    }

    SubdomainTopology<dim, spacedim> &
    get_subdomain_topology()
    {
      return subdomain_topology;
    }

  private:
    SubdomainTopology<dim, spacedim>     subdomain_topology;
    EfieldSubdomainDescription<spacedim> domain;
    Triangulation<dim, spacedim>         tria;
    /**
     * This @p std::map associates mapping orders and mapping objects.
     */
    std::map<unsigned int, MappingInfo<dim, spacedim> *> mappings;

    /**
     * Dirichlet space on the whole skeleton.
     */
    DoFHandler<dim, spacedim> dof_handler_for_dirichlet_space;
    /**
     * Neumann space on the whole skeleton.
     */
    DoFHandler<dim, spacedim> dof_handler_for_neumann_space;

    /**
     * Finite element order for the Dirichlet space.
     */
    unsigned int fe_order_for_dirichlet_space;

    /**
     * Finite element order for the Neumann space.
     */
    unsigned int fe_order_for_neumann_space;

    /**
     * Finite element \f$H^{\frac{1}{2}+s}\f$ for the Dirichlet space. At
     * present, it is implemented as a continuous Lagrange space.
     */
    FE_Q<dim, spacedim> fe_for_dirichlet_space;
    /**
     * Finite element \f$H^{-\frac{1}{2}+s}\f$ for the Neumann space. At
     * present, it is implemented as a discontinuous Lagrange space.
     */
    FE_DGQ<dim, spacedim> fe_for_neumann_space;

    DDMEfieldMatrix<spacedim, double>               system_matrix;
    DDMEfieldGlobalPreconditioner<spacedim, double> system_preconditioner;
    Vector<double>                                  rhs_for_transmission_eqn;
    Vector<double> rhs_for_charge_neutrality_eqn;

    /**
     * Dirichlet boundary condition data on all DoFs in the associated DoF
     * handler.
     */
    Vector<double> dirichlet_bc;

    /**
     * Map volume entity tag to subdomain type.
     */
    std::map<EntityTag, SubdomainType> subdomain_types;
    /**
     * Map volume entity tag to permittivity.
     */
    std::map<EntityTag, double> permittivities;
    /**
     * Map volume entity tag to voltages of conductors.
     */
    std::map<EntityTag, double> conductor_voltages;
    /**
     * Map surface entity tag to Dirichlet boundary condition.
     */
    std::map<EntityTag, Function<spacedim, double> *>
      dirichlet_boundary_conditions;
    /**
     * Map surface entity tag to manifold id. At the moment, the material for
     * each surface is the same as the entity tag in Gmsh.
     */
    std::map<EntityTag, types::manifold_id> manifold_description;
    /**
     * Map @p manifold_id to the pointer of a Manifold object.
     */
    std::map<types::manifold_id, Manifold<dim, spacedim> *> manifolds;
    /**
     * Map @p manifold_id to mapping order.
     */
    std::map<types::manifold_id, unsigned int> manifold_id_to_mapping_order;
  };


  // Declaration of deal.ii internal functions which will be used in
  // DDMEfield<dim, spacedim>::read_mesh.
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


  template <int dim, int spacedim>
  DDMEfield<dim, spacedim>::DDMEfield()
    : fe_order_for_dirichlet_space(1)
    , fe_order_for_neumann_space(0)
    , fe_for_dirichlet_space(fe_order_for_dirichlet_space)
    , fe_for_neumann_space(fe_order_for_neumann_space)
  {}


  template <int dim, int spacedim>
  DDMEfield<dim, spacedim>::~DDMEfield()
  {
    // Release function objects defining Dirichlet boundary conditions on
    // dielectric surfaces or interfaces.
    for (auto &d : dirichlet_boundary_conditions)
      {
        delete d.second;
      }

    // Release mapping information objects.
    for (auto &m : mappings)
      {
        delete m.second;
      }

    // Release manifold objects.
    for (auto &m : manifolds)
      {
        delete m.second;
      }
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::read_skeleton_mesh(const std::string &mesh_file)
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

  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::read_subdomain_topology(
    const std::string &cad_file,
    const std::string &mesh_file)
  {
    subdomain_topology.generate_topology(cad_file, mesh_file);
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::initialize_parameters()
  {
    subdomain_types[0] = SubdomainType::SurroundingSpace;
    subdomain_types[1] = SubdomainType::VoltageConductor;
    subdomain_types[2] = SubdomainType::Dielectric;
    subdomain_types[3] = SubdomainType::Dielectric;

    permittivities[0] = 1;
    permittivities[2] = 2;
    permittivities[3] = 4;

    // Assign Dirichlet boundary condition on dielectric boundary or interface.
    conductor_voltages[1] = 10;

    std::string                   symbolic_variables = "x,y,z";
    std::map<std::string, double> symbolic_constants;
    symbolic_constants["pi"] = numbers::PI;
    symbolic_constants["e"]  = numbers::E;

    dirichlet_boundary_conditions[11] = new FunctionParser<spacedim>(1);
    std::string expr                  = "sin(x) * cos(z)";
    static_cast<FunctionParser<spacedim> *>(dirichlet_boundary_conditions[11])
      ->initialize(symbolic_variables, expr, symbolic_constants);

    dirichlet_boundary_conditions[6] = new FunctionParser<spacedim>(1);
    expr                             = "cos(x) * sin(z)";
    static_cast<FunctionParser<spacedim> *>(dirichlet_boundary_conditions[6])
      ->initialize(symbolic_variables, expr, symbolic_constants);

    // Spherical manifold
    manifold_description[1] = 1;
    manifold_description[2] = 1;

    // Flat manifold
    for (unsigned int i = 3; i <= 13; i++)
      {
        manifold_description[i] = 0;
      }

    // We use the first order mapping for the flat manifold and the second order
    // mapping for the spherical manifold.
    manifold_id_to_mapping_order[0] = 1;
    manifold_id_to_mapping_order[1] = 2;
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::initialize_manifolds_and_mappings()
  {
    // Assign manifold ids to all cells in the triangulation. Because there are
    // no physical groups defined, the manifold ids cannot be read from the MSH
    // file and have to be initialized here.
    for (auto &cell : tria.active_cell_iterators())
      {
        cell->set_all_manifold_ids(manifold_description[cell->material_id()]);
      }

    // Create and associate manifold objects.
    for (const auto &desc : manifold_description)
      {
        switch (desc.second)
          {
              case 1: {
                typename decltype(manifolds)::iterator pos =
                  manifolds.find(desc.second);

                if (pos == manifolds.end())
                  {
                    // Define a 2D spherical manifold centered at the origin, if
                    // it has not been defined before.
                    manifolds[desc.second] =
                      new SphericalManifold<dim, spacedim>(
                        Point<spacedim>(0., 0., 0.));
                    tria.set_manifold(desc.second, *manifolds[desc.second]);
                  }

                break;
              }
          }
      }

    // Create mapping objects.
    for (const auto &record : manifold_id_to_mapping_order)
      {
        typename decltype(mappings)::iterator pos =
          mappings.find(record.second);

        if (pos == mappings.end())
          {
            // Create the mapping object with the specified order, if it has not
            // been created before.
            mappings[record.second] =
              new MappingInfo<dim, spacedim>(record.second);
          }
      }
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::interpolate_surface_dirichlet_bc()
  {
    dirichlet_bc.reinit(dof_handler_for_dirichlet_space.n_dofs());

    // Because each surface may be assigned a different mapping object, here we
    // interpolate the Dirichlet boundary condition vector surface by surface.
    for (const auto &bc : dirichlet_boundary_conditions)
      {
        std::map<types::material_id, const Function<spacedim, double> *>
          single_pair_map;
        single_pair_map[static_cast<types::material_id>(bc.first)] = bc.second;

        VectorTools::interpolate_based_on_material_id(
          mappings[manifold_id_to_mapping_order[manifold_description[bc.first]]]
            ->get_mapping(),
          dof_handler_for_dirichlet_space,
          single_pair_map,
          dirichlet_bc);
      }
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::create_efield_subdomains_and_surfaces()
  {
    // Create the default surrounding space subdomain (e.g. air domain) and add
    // it as the first dielectric subdomain in the domain description object.
    domain.get_subdomains()[0] = EfieldSubdomain<spacedim>(
      0, SubdomainType::SurroundingSpace, permittivities[0], 0);
    domain.get_dielectric_subdomains().push_back(&domain.get_subdomains()[0]);

    // Create each subdomain.
    for (const auto &record : subdomain_topology.get_subdomain_to_surface())
      {
        const EntityTag     entity_tag = record.first;
        const SubdomainType type       = subdomain_types[record.first];
        const double        permittivity =
          (type == SubdomainType::Dielectric) ? permittivities[entity_tag] : 0;
        const double voltage = (type == SubdomainType::VoltageConductor) ?
                                 conductor_voltages[entity_tag] :
                                 0;

        domain.get_subdomains()[entity_tag] =
          EfieldSubdomain<spacedim>(entity_tag, type, permittivity, voltage);

        // Add the subdomain to corresponding list in the domain description
        // object.
        switch (type)
          {
            case (SubdomainType::Dielectric):
              domain.get_dielectric_subdomains().push_back(
                &domain.get_subdomains()[entity_tag]);
              break;
            case (SubdomainType::VoltageConductor):
              domain.get_voltage_conductor_subdomains().push_back(
                &domain.get_subdomains()[entity_tag]);
              break;
            case (SubdomainType::FloatingConductor):
              domain.get_floating_conductor_subdomains().push_back(
                &domain.get_subdomains()[entity_tag]);
              break;
            default:
              Assert(false, ExcInternalError());
              break;
          }
      }

    // Create each surface. For a same geometric surface, there are actually two
    // @p EfieldSurface objects are created and associated.
    for (const auto &record : subdomain_topology.get_surface_to_subdomain())
      {
        // Determine if the current surface is assigned a Dirichlet boundary
        // condition.
        typename std::map<EntityTag, Function<spacedim, double> *>::iterator
             pos = dirichlet_boundary_conditions.find(record.first);
        bool is_dirichlet_surface;
        Function<spacedim, double> *dirichlet_voltage;

        if (pos != dirichlet_boundary_conditions.end())
          {
            is_dirichlet_surface = true;
            dirichlet_voltage    = pos->second;
          }
        else
          {
            is_dirichlet_surface = false;
            dirichlet_voltage    = nullptr;
          }

        // Get the two subdomains sharing the current surface.
        EfieldSubdomain<spacedim> &subdomain_surface_normal_point_from =
          domain.get_subdomains()[record.second[0]];
        EfieldSubdomain<spacedim> &subdomain_surface_normal_point_to =
          domain.get_subdomains()[record.second[1]];

        EfieldSurface<spacedim> *surface_of_from_subdomain;
        EfieldSurface<spacedim> *surface_of_to_subdomain;

        // Create a surface object for the "from" subdomain.
        // Here we check its neighboring subdomain type and add the surface to
        // corresponding list in the current subdomain.
        switch (subdomain_surface_normal_point_to.get_type())
          {
            case SubdomainType::SurroundingSpace:
              case SubdomainType::Dielectric: {
                subdomain_surface_normal_point_from
                  .get_surfaces_touching_dielectric()
                  .emplace_back(record.first,
                                nullptr, // Temporarily, the neighbor surface is
                                         // not connected.
                                &subdomain_surface_normal_point_from,
                                true,
                                is_dirichlet_surface,
                                dirichlet_voltage);
                surface_of_from_subdomain =
                  &subdomain_surface_normal_point_from
                     .get_surfaces_touching_dielectric()
                     .back();
                break;
              }
              case SubdomainType::VoltageConductor: {
                subdomain_surface_normal_point_from
                  .get_surfaces_touching_voltage_conductor()
                  .emplace_back(record.first,
                                nullptr, // Temporarily, the neighbor surface is
                                         // not connected.
                                &subdomain_surface_normal_point_from,
                                true,
                                is_dirichlet_surface,
                                dirichlet_voltage);
                surface_of_from_subdomain =
                  &subdomain_surface_normal_point_from
                     .get_surfaces_touching_voltage_conductor()
                     .back();
                break;
              }
              case SubdomainType::FloatingConductor: {
                subdomain_surface_normal_point_from
                  .get_surfaces_touching_floating_conductor()
                  .emplace_back(record.first,
                                nullptr, // Temporarily, the neighbor surface is
                                         // not connected.
                                &subdomain_surface_normal_point_from,
                                true,
                                is_dirichlet_surface,
                                dirichlet_voltage);
                surface_of_from_subdomain =
                  &subdomain_surface_normal_point_from
                     .get_surfaces_touching_floating_conductor()
                     .back();
                break;
              }
          }

        // Create a surface object for the "to" subdomain.
        // Here we check its neighboring subdomain type and add the surface to
        // corresponding list in the current subdomain.
        switch (subdomain_surface_normal_point_from.get_type())
          {
            case SubdomainType::SurroundingSpace:
              case SubdomainType::Dielectric: {
                subdomain_surface_normal_point_to
                  .get_surfaces_touching_dielectric()
                  .emplace_back(record.first,
                                nullptr, // Temporarily, the neighbor surface is
                                         // not connected.
                                &subdomain_surface_normal_point_to,
                                false,
                                is_dirichlet_surface,
                                dirichlet_voltage);
                surface_of_to_subdomain = &subdomain_surface_normal_point_to
                                             .get_surfaces_touching_dielectric()
                                             .back();
                break;
              }
              case SubdomainType::VoltageConductor: {
                subdomain_surface_normal_point_to
                  .get_surfaces_touching_voltage_conductor()
                  .emplace_back(record.first,
                                nullptr, // Temporarily, the neighbor surface is
                                         // not connected.
                                &subdomain_surface_normal_point_to,
                                false,
                                is_dirichlet_surface,
                                dirichlet_voltage);
                surface_of_to_subdomain =
                  &subdomain_surface_normal_point_to
                     .get_surfaces_touching_voltage_conductor()
                     .back();
                break;
              }
              case SubdomainType::FloatingConductor: {
                subdomain_surface_normal_point_to
                  .get_surfaces_touching_floating_conductor()
                  .emplace_back(record.first,
                                nullptr, // Temporarily, the neighbor surface is
                                         // not connected.
                                &subdomain_surface_normal_point_to,
                                false,
                                is_dirichlet_surface,
                                dirichlet_voltage);
                surface_of_to_subdomain =
                  &subdomain_surface_normal_point_to
                     .get_surfaces_touching_floating_conductor()
                     .back();
                break;
              }
          }

        // Connect neighboring surfaces.
        surface_of_from_subdomain->set_neighbor_surface(
          surface_of_to_subdomain);
        surface_of_to_subdomain->set_neighbor_surface(
          surface_of_from_subdomain);
      }
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::initialize_subdomain_hmatrices()
  {
    auto &subdomain_hmatrices = system_matrix.get_subdomain_hmatrices();
    // TODO 2024-08-10 At the moment, the memory should be reserved for this
    // vector to prevent std::vector copying or moving data, because this class
    // has no copy or move-copy constructor defined yet.
    subdomain_hmatrices.reserve(domain.get_dielectric_subdomains().size());

    // Iterate over each dielectric subdomain.
    std::set<types::material_id> material_ids;
    for (const auto subdomain : domain.get_dielectric_subdomains())
      {
        // Create an empty Steklov_poincare \hmat for the current subdomain.
        subdomain_hmatrices.emplace_back();

        // Generate selectors for Dirichlet space.
        std::vector<bool> dof_selectors_for_dirichlet_space;
        dof_selectors_for_dirichlet_space.resize(
          dof_handler_for_dirichlet_space.n_dofs());

        material_ids.clear();
        for (const auto &surface :
             subdomain->get_surfaces_touching_dielectric())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }

        for (const auto &surface :
             subdomain->get_surfaces_touching_floating_conductor())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }

        for (const auto &surface :
             subdomain->get_surfaces_touching_voltage_conductor())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }

        DoFToolsExt::extract_material_domain_dofs(
          dof_handler_for_dirichlet_space,
          material_ids,
          dof_selectors_for_dirichlet_space);

        // Generate selectors for Neumann space.
        std::vector<bool> dof_selectors_for_neumann_space;
        dof_selectors_for_neumann_space.resize(
          dof_handler_for_neumann_space.n_dofs());

        material_ids.clear();
        for (const auto &surface :
             subdomain->get_surfaces_touching_dielectric())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }

        for (const auto &surface :
             subdomain->get_surfaces_touching_floating_conductor())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }

        for (const auto &surface :
             subdomain->get_surfaces_touching_voltage_conductor())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }

        DoFToolsExt::extract_material_domain_dofs(
          dof_handler_for_neumann_space,
          material_ids,
          dof_selectors_for_neumann_space);

        subdomain_hmatrices.back().build_local_to_global_dof_maps(
          dof_handler_for_dirichlet_space,
          dof_handler_for_neumann_space,
          dof_selectors_for_dirichlet_space,
          dof_selectors_for_neumann_space);

        // Get the number of effective DoF number for each DoF handler.
        const unsigned int n_dofs_for_local_dirichlet_space =
          subdomain_hmatrices.back()
            .get_subdomain_to_skeleton_dirichlet_dof_index_map()
            .size();
        const unsigned int n_dofs_for_local_neumann_space =
          subdomain_hmatrices.back()
            .get_subdomain_to_skeleton_neumann_dof_index_map()
            .size();

        std::cout << "=== DoF information on subdomain "
                  << subdomain->get_entity_tag() << "===" << std::endl;
        std::cout << "Number of DoFs in local Dirichlet space: "
                  << n_dofs_for_local_dirichlet_space << "\n";
        std::cout << "Number of DoFs in local Neumann space: "
                  << n_dofs_for_local_neumann_space << std::endl;
      }

    // Generate selectors for Dirichlet space on non-Dirichlet boundary.
    std::vector<bool>
      negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary;
    negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary.resize(
      dof_handler_for_dirichlet_space.n_dofs());

    material_ids.clear();
    for (const auto subdomain : domain.get_voltage_conductor_subdomains())
      {
        for (const auto &surface :
             subdomain->get_surfaces_touching_dielectric())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }
      }

    for (const auto subdomain : domain.get_floating_conductor_subdomains())
      {
        for (const auto &surface :
             subdomain->get_surfaces_touching_dielectric())
          {
            material_ids.insert(
              static_cast<types::material_id>(surface.get_entity_tag()));
          }
      }

    for (const auto subdomain : domain.get_dielectric_subdomains())
      {
        for (const auto &surface :
             subdomain->get_surfaces_touching_dielectric())
          {
            if (surface.get_is_dirichlet_boundary())
              {
                material_ids.insert(
                  static_cast<types::material_id>(surface.get_entity_tag()));
              }
          }
      }

    DoFToolsExt::extract_material_domain_dofs(
      dof_handler_for_dirichlet_space,
      material_ids,
      negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary);

    system_matrix
      .build_local_to_global_dirichlet_dof_map_on_nondirichlet_boundary(
        dof_handler_for_dirichlet_space,
        negated_dof_selectors_for_dirichlet_space_on_nondirichlet_boundary);

    std::cout
      << "Number of DoFs in local Dirichlet space restricted to non-Dirichlet boundary: "
      << system_matrix
           .get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map()
           .size()
      << std::endl;
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::setup_system()
  {
    // Create subdomain and surface objects.
    create_efield_subdomains_and_surfaces();

    initialize_manifolds_and_mappings();

    //#if ENABLE_DEBUG == 1
    //    // Refine the mesh to see if the assigned spherical manifold works.
    //    tria.refine_global(1);
    //    // Write out the mesh to check if elementary tags have been assigned
    //    // to material ids.
    //    std::ofstream out("output.msh");
    //    write_msh_correct<dim, spacedim>(tria, out);
    //    out.close();
    //#endif

    // Initialize DoF handlers.
    dof_handler_for_dirichlet_space.reinit(tria);
    dof_handler_for_dirichlet_space.distribute_dofs(fe_for_dirichlet_space);
    dof_handler_for_neumann_space.reinit(tria);
    dof_handler_for_neumann_space.distribute_dofs(fe_for_neumann_space);

    // Apply Dirichlet boundary conditions on dielectric surfaces.
    interpolate_surface_dirichlet_bc();

    initialize_subdomain_hmatrices();
  }


  template <int dim, int spacedim>
  void
  DDMEfield<dim, spacedim>::output_results()
  {
    // Write the Dirichlet boundary condition.
    std::ofstream          vtk_output("out.vtk");
    DataOut<dim, spacedim> data_out;
    data_out.add_data_vector(dof_handler_for_dirichlet_space,
                             dirichlet_bc,
                             "dirichlet_bc");

    Vector<double> dofs_on_nondirichlet_boundary;
    dofs_on_nondirichlet_boundary.reinit(
      dof_handler_for_dirichlet_space.n_dofs());
    for (auto index :
         system_matrix
           .get_nondirichlet_boundary_to_skeleton_dirichlet_dof_index_map())
      {
        dofs_on_nondirichlet_boundary[index] = 1.0;
      }
    data_out.add_data_vector(dof_handler_for_dirichlet_space,
                             dofs_on_nondirichlet_boundary,
                             "nondirichlet");

    data_out.build_patches();
    data_out.write_vtk(vtk_output);
  }
} // namespace HierBEM


#endif /* INCLUDE_ELECTRIC_FIELD_DDM_EFIELD_H_ */
