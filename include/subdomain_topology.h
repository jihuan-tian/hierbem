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
     * They are logically considered to be a single domain.
     *
     * @pre
     * @post
     */
    void
    generate_single_domain_topology_for_dealii_model(
      const std::vector<EntityTag> &surface_list);

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

    // Check if the format of the CAD file is geo.
    size_t is_geo_cad = cad_file.rfind(std::string(".geo"));

    if (is_geo_cad != std::string::npos)
      {
        if (dim > 1)
          throw(ExcMessage("geo model can only be used for 2D model!"));
        else
          gmsh::model::geo::synchronize();
      }
    else
      gmsh::model::occ::synchronize();

    Assert(gmsh::model::getDimension() >= dim + 1, ExcInternalError());

    gmsh::vectorpair volume_dimtag_list;
    if (is_geo_cad)
      gmsh::model::getEntities(volume_dimtag_list, dim + 1);
    else
      gmsh::model::occ::getEntities(volume_dimtag_list, dim + 1);

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
    generate_single_domain_topology_for_dealii_model(
      const std::vector<EntityTag> &surface_list)
  {
    std::array<EntityTag, 2> subdomains{{1, 0}};
    for (auto s : surface_list)
      {
        subdomain_to_surface[1].push_back(s);
        surface_to_subdomain[s] = subdomains;
      }
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
