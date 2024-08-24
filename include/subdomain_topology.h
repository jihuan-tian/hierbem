/**
 * @file subdomain_topology.h
 * @brief Introduction of subdomain_topology.h
 *
 * @date 2024-08-15
 * @author Jihuan Tian
 */
#ifndef INCLUDE_SUBDOMAIN_TOPOLOGY_H_
#define INCLUDE_SUBDOMAIN_TOPOLOGY_H_

#include <deal.II/base/exceptions.h>

#include <gmsh.h>

#include <array>
#include <iostream>
#include <map>

#include "gmsh_manipulation.h"

namespace HierBEM
{
  using namespace dealii;

  template <int dim, int spacedim>
  class SubdomainTopology
  {
  public:
    SubdomainTopology() = default;

    void
    generate_topology(const std::string &cad_file,
                      const std::string &mesh_file,
                      const double       eps_for_orientation_detection = 1e-5);

    /**
     * Generate a default topology for a single domain created in deal.ii, whose
     * surface normals all point outward and surfaces are not assigned any
     * physical groups.
     *
     * All cells on the the surfaces have zero material ids. Even though there
     * may be several bodies created in deal.ii, they are logically considered
     * to be a single domain.
     *
     * @pre
     * @post
     */
    void
    generate_single_domain_topology_for_dealii_model();

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
  SubdomainTopology<dim, spacedim>::
    generate_single_domain_topology_for_dealii_model()
  {
    // Actually, there is only one logical surface with entity tag 0.
    std::vector<EntityTag> surface_list{0};
    subdomain_to_surface[1] = surface_list;

    // There is only one logical volume in the domain, and the surface normals
    // point outward.
    std::array<EntityTag, 2> subdomains{{1, 0}};
    surface_to_subdomain[0] = subdomains;
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
} // namespace HierBEM

#endif /* INCLUDE_SUBDOMAIN_TOPOLOGY_H_ */
