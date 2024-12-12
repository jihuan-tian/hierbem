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


  /**
   * A class for detecting if a surface normal vector points into a volume.
   *
   * If so, the surface normal vector computed for a cell should be negated,
   * because we assume an outward normal vector adopted for the problem
   * domain, whether we're solving an interior BEM problem or exterior
   * problem.
   */
  template <int dim, int spacedim>
  class SurfaceNormalDetector
  {
  public:
    SurfaceNormalDetector() = delete;

    SurfaceNormalDetector(SubdomainTopology<dim, spacedim> &subdomain_topology)
      : subdomain_topology(subdomain_topology)
    {}

    /**
     * Given a material id of a cell, this function checks if its normal
     * vector points into a corresponding domain by checking the
     * surface-to-subdomain relationship.
     *
     * \mynote{In the Laplace solver, a domain (with a non-zero subdomain tag)
     * must be fully in contact with the surrounding space (whose subdomain
     * tag is zero). This still holds if there are several subdomains in the
     * model, because they are all well separated from each other. This leads
     * to the fact the in a record in the surface-to-subdomain relationship,
     * there should be only one non-zero value. We use this fact to check the
     * direction of the surface normal vector.}
     *
     * @pre
     * @post
     * @param m
     * @return
     */
    bool
    is_normal_vector_inward(const types::material_id m) const
    {
      if (subdomain_topology.get_surface_to_subdomain()[m][0] > 0)
        {
          Assert(subdomain_topology.get_surface_to_subdomain()[m][1] == 0,
                 ExcInternalError());
          return false;
        }
      else
        {
          Assert(subdomain_topology.get_surface_to_subdomain()[m][1] > 0,
                 ExcInternalError());
          return true;
        }
    }

  private:
    SubdomainTopology<dim, spacedim> &subdomain_topology;
  };
} // namespace HierBEM

#endif /* INCLUDE_SUBDOMAIN_TOPOLOGY_H_ */
