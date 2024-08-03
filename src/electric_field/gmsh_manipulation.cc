/**
 * @file gmsh_mainpulation.cc
 * @brief Introduction of gmsh_mainpulation.cc
 *
 * @date 2024-08-01
 * @author Jihuan Tian
 */

#include "electric_field/gmsh_manipulation.h"

#include <deal.II/base/exceptions.h>

#include <vector>

namespace HierBEM
{
  using namespace dealii;

  bool
  GmshManip::is_point_in_volume(const EntityTag            volume_tag,
                                const std::vector<double> &coord)
  {
    if (gmsh::model::isInside(3, volume_tag, coord, false) > 0)
      {
        return true;
      }
    else
      {
        return false;
      }
  }

  void
  GmshManip::get_surface_normal_at_cell_barycenter(
    const EntityTag      surface_tag,
    std::vector<double> &normal,
    std::vector<double> &root_point)
  {
    std::vector<int>                      element_types;
    std::vector<std::vector<std::size_t>> element_tags;
    std::vector<std::vector<std::size_t>> node_tags;

    gmsh::model::mesh::getElements(
      element_types, element_tags, node_tags, 2, surface_tag);

    // Ensure that we only have one type of elements.
    AssertDimension(element_types.size(), 1);

    // Get barycenters of all cells on the surface.
    std::vector<double> barycenters;
    gmsh::model::mesh::getBarycenters(
      element_types[0], surface_tag, false, false, barycenters);

    // We only use the first barycenter to get the normal vector.
    std::vector<double> first_barycenter(3);
    first_barycenter[0] = barycenters[0];
    first_barycenter[1] = barycenters[1];
    first_barycenter[2] = barycenters[2];

    std::vector<double> uv(2);
    gmsh::model::getParametrization(2, surface_tag, first_barycenter, uv);
    gmsh::model::getNormal(surface_tag, uv, normal);
    // Because the cell is flat while the surface may be curved, the barycenter
    // may not lie exactly on the surface. Hence, we transform the uv local
    // coordinates to Cartesian frame, which inherently perform projection onto
    // the surface.
    gmsh::model::getValue(2, surface_tag, uv, root_point);
  }


  int
  GmshManip::get_surface_intrinsic_orientation(const EntityTag surface_tag,
                                               const EntityTag volume_tag,
                                               const double    eps)
  {
    std::vector<double> normal(3);
    std::vector<double> root_point(3);

    get_surface_normal_at_cell_barycenter(surface_tag, normal, root_point);
    // Translate the root point of the normal vector along the normal direction
    // a little bit. If the translated point is within the volume, the normal
    // vector is inward, which has the orientation -1. If the point is outside
    // the volume, the normal vector is outward, which has the orientation +1.
    add_vector(root_point, normal, eps);

    return is_point_in_volume(volume_tag, root_point) ? -1 : 1;
  }


  void
  GmshManip::add_vector(std::vector<double>       &x,
                        const std::vector<double> &y,
                        double                     alpha)
  {
    AssertDimension(x.size(), y.size());

    for (unsigned int i = 0; i < x.size(); i++)
      {
        x[i] += alpha * y[i];
      }
  }


  void
  GmshManip::get_oriented_volume_boundaries(
    const EntityTag         volume_tag,
    std::vector<EntityTag> &oriented_surface_tags,
    const double            eps)
  {
    gmsh::vectorpair volume_dimtag(1);
    volume_dimtag[0].first  = 3;
    volume_dimtag[0].second = volume_tag;

    gmsh::vectorpair surface_dimtag_list;
    gmsh::model::getBoundary(volume_dimtag, surface_dimtag_list, false, false);

    oriented_surface_tags.clear();
    for (const auto &surface_dimtag : surface_dimtag_list)
      {
        oriented_surface_tags.push_back(
          surface_dimtag.second * get_surface_intrinsic_orientation(
                                    surface_dimtag.second, volume_tag, eps));
      }
  }


  void
  GmshManip::get_oriented_volume_boundaries(
    const EntityTag                                volume_tag,
    std::vector<EntityTag>                        &oriented_surface_tags,
    std::map<EntityTag, std::array<EntityTag, 2>> &face_to_subdomain,
    const double                                   eps)
  {
    gmsh::vectorpair volume_dimtag(1);
    volume_dimtag[0].first  = 3;
    volume_dimtag[0].second = volume_tag;

    gmsh::vectorpair surface_dimtag_list;
    gmsh::model::getBoundary(volume_dimtag, surface_dimtag_list, false, false);

    oriented_surface_tags.clear();
    for (const auto &surface_dimtag : surface_dimtag_list)
      {
        int surface_orientation;

        // Here we first check if this surface has been met before.
        typename std::map<EntityTag, std::array<EntityTag, 2>>::iterator pos =
          face_to_subdomain.find(surface_dimtag.second);
        if (pos != face_to_subdomain.end())
          {
            if (pos->second[0] != 0 && pos->second[1] == 0)
              {
                surface_orientation = -1;
                pos->second[1]      = volume_tag;
              }
            else if (pos->second[0] == 0 && pos->second[1] != 0)
              {
                surface_orientation = 1;
                pos->second[0]      = volume_tag;
              }
            else
              {
                // This case cannot happen.
                Assert(false, ExcInternalError());
              }
          }
        else
          {
            surface_orientation =
              get_surface_intrinsic_orientation(surface_dimtag.second,
                                                volume_tag,
                                                eps);

            // Insert a new record in @p face_to_subdomain map.
            if (surface_orientation == 1)
              {
                face_to_subdomain[surface_dimtag.second] = {{volume_tag, 0}};
              }
            else
              {
                face_to_subdomain[surface_dimtag.second] = {{0, volume_tag}};
              }
          }

        oriented_surface_tags.push_back(surface_dimtag.second *
                                        surface_orientation);
      }
  }
} // namespace HierBEM
