/**
 * @file gmsh_manipulation.h
 * @brief Introduction of gmsh_manipulation.h
 *
 * @date 2024-08-01
 * @author Jihuan Tian
 */
#ifndef INCLUDE_ELECTRIC_FIELD_GMSH_MANIPULATION_H_
#define INCLUDE_ELECTRIC_FIELD_GMSH_MANIPULATION_H_

#include <gmsh.h>

#include <array>
#include <map>
#include <vector>

#include "vector_arithmetic.h"

namespace HierBEM
{
  using EntityTag = int;

  /**
   * Geometric and mesh manipulation using Gmsh API.
   *
   * @tparam dim Dimension of the BEM domain.
   * @tparam spacedim
   */
  template <int dim, int spacedim>
  class GmshManip
  {
  public:
    /**
     * Check if a point with the given Cartesian coordinates is inside a
     * volume.
     *
     * @pre
     * @post
     * @param entity_tag Entity tag of the subdomain, which has one more
     * dimension than the BEM domain of dimension @p dim.
     * @param coord Cartesian coordinates of dimension @p spacedim.
     * @return
     */
    static bool
    is_point_in_volume(const EntityTag            volume_tag,
                       const std::vector<double> &coord);

    static void
    get_normal(const EntityTag            surface_tag,
               const std::vector<double> &parametric_coord,
               std::vector<double>       &normal);

    /**
     * Get the normal vector of a cell on the specified surface.
     *
     * \mynote{When @p dim=1, the surface is actually an edge.}
     *
     * The root point of the normal vector is at a point on the surface, which
     * is a projection of the barycenter of the cell. Since the cell is flat
     * while the surface is curved, the barycenter may not lie exactly on the
     * surface.
     *
     * @pre
     * @post
     * @param surface_tag
     * @param normal
     * @param root_point
     */
    static void
    get_surface_normal_at_cell_barycenter(const EntityTag      surface_tag,
                                          std::vector<double> &normal,
                                          std::vector<double> &root_point);

    /**
     * Get the intrinsic orientation of the surface.
     *
     * \mynote{This intrinsic orientation should be differentiated from the
     * topological orientation of the surface with respect to its attached
     * volume.
     *
     * When @p dim=2, the former is determined by the local coordinate chart
     * \f$(u,v)\f$ assigned to the surface, while the latter is determined by
     * the orientation (with respect to the volume) of the curve loop, which
     * encloses the surface.
     *
     * The orientation is computed by checking that if a point is inside the
     * volume, which starts from a surface point and translates along the normal
     * vector with an incremental distance. If the point lies within the volume,
     * the surface orientation is negative, otherwise it is positive.}
     *
     * @pre
     * @post
     * @param surface_tag
     * @param volume_tag
     * @param eps The incremental distance by which the point moves along the
     * normal vector, which is used to detect orientation.
     * @return If the surface normal vector is outward with respect to the
     * volume, the orientation is 1. If the normal vector points inward, the
     * orientation is -1.
     */
    static int
    get_surface_intrinsic_orientation(const EntityTag surface_tag,
                                      const EntityTag volume_tag,
                                      const double    eps = 1e-5);

    /**
     * Get a signed tag list for the surfaces constituting the boundary of a
     * volume. A positive tag represents surface normal vector points outward,
     * while a negative tag represents an inward normal vector.
     *
     * @pre
     * @post
     * @param volume_tag
     * @param oriented_surface_tags
     * @param eps
     */
    static void
    get_oriented_volume_boundaries(
      const EntityTag         volume_tag,
      std::vector<EntityTag> &oriented_surface_tags,
      const double            eps = 1e-5);

    /**
     * By checking if a surface has been met before, this overloaded version
     * reduce operations of getting barycenters of cells on a surface.
     * Meanwhile, the face-to-subdomain relationship is also built.
     *
     * @pre
     * @post
     * @param volume_tag
     * @param oriented_surface_tags
     * @param face_to_subdomain
     * @param eps
     */
    static void
    get_oriented_volume_boundaries(
      const EntityTag                                volume_tag,
      std::vector<EntityTag>                        &oriented_surface_tags,
      std::map<EntityTag, std::array<EntityTag, 2>> &face_to_subdomain,
      const double                                   eps = 1e-5);
  };


  template <int dim, int spacedim>
  bool
  GmshManip<dim, spacedim>::is_point_in_volume(const EntityTag volume_tag,
                                               const std::vector<double> &coord)
  {
    AssertDimension(coord.size(), spacedim);

    if (dim == 1)
      {
        for (unsigned int i = dim + 1; i < spacedim; i++)
          Assert(coord[i] == 0, ExcInternalError());
      }

    if (gmsh::model::isInside(dim + 1, volume_tag, coord, false) > 0)
      {
        return true;
      }
    else
      {
        return false;
      }
  }


  template <int dim, int spacedim>
  void
  GmshManip<dim, spacedim>::get_normal(
    const EntityTag            entity_tag,
    const std::vector<double> &parametric_coord,
    std::vector<double>       &normal)
  {
    AssertDimension(normal.size(), spacedim);
    AssertDimension(parametric_coord.size(), dim);

    switch (dim)
      {
          case 1: {
            // Compute the tangent vector on the curve.
            std::vector<double> tangent;
            gmsh::model::getDerivative(dim,
                                       entity_tag,
                                       parametric_coord,
                                       tangent);

            for (unsigned int i = dim + 1; i < spacedim; i++)
              Assert(tangent[i] == 0, ExcInternalError());

            // Normalize the tangent vector.
            normalize_vector(tangent);

            // Rotate the tangent vector by \f$-\frac{\pi}{2}\f$ to get the
            // normal vector.
            normal[0] = tangent[1];
            normal[1] = -tangent[0];
            for (unsigned int i = dim + 1; i < spacedim; i++)
              normal[i] = 0;

            break;
          }
          case 2: {
            gmsh::model::getNormal(entity_tag, parametric_coord, normal);
            break;
          }
          default: {
            Assert(false, ExcInternalError());
            break;
          }
      }
  }


  template <int dim, int spacedim>
  void
  GmshManip<dim, spacedim>::get_surface_normal_at_cell_barycenter(
    const EntityTag      surface_tag,
    std::vector<double> &normal,
    std::vector<double> &root_point)
  {
    AssertDimension(normal.size(), spacedim);
    AssertDimension(root_point.size(), spacedim);

    std::vector<int>                      element_types;
    std::vector<std::vector<std::size_t>> element_tags;
    std::vector<std::vector<std::size_t>> node_tags;

    gmsh::model::mesh::getElements(
      element_types, element_tags, node_tags, dim, surface_tag);

    // Ensure that we only have one type of elements.
    AssertDimension(element_types.size(), 1);

    // Get barycenters of all cells on the surface.
    std::vector<double> barycenters;
    gmsh::model::mesh::getBarycenters(
      element_types[0], surface_tag, false, false, barycenters);

    // We only use the first barycenter to get the normal vector.
    std::vector<double> first_barycenter(spacedim);
    for (unsigned int i = 0; i < spacedim; i++)
      {
        first_barycenter[i] = barycenters[i];
      }

    if (dim == 1)
      {
        for (unsigned int i = dim + 1; i < spacedim; i++)
          Assert(first_barycenter[i] == 0, ExcInternalError());
      }

    std::vector<double> parametric_coord(dim);
    gmsh::model::getParametrization(dim,
                                    surface_tag,
                                    first_barycenter,
                                    parametric_coord);
    get_normal(surface_tag, parametric_coord, normal);

    // Because the cell is flat while the surface may be curved, the barycenter
    // may not lie exactly on the surface. Hence, we transform the t (for
    // @p dim=1) or uv (for @p dim=2) local coordinates to Cartesian frame,
    // which inherently perform projection onto the surface.
    gmsh::model::getValue(dim, surface_tag, parametric_coord, root_point);
  }


  template <int dim, int spacedim>
  int
  GmshManip<dim, spacedim>::get_surface_intrinsic_orientation(
    const EntityTag surface_tag,
    const EntityTag volume_tag,
    const double    eps)
  {
    std::vector<double> normal(spacedim);
    std::vector<double> root_point(spacedim);

    get_surface_normal_at_cell_barycenter(surface_tag, normal, root_point);
    // Translate the root point of the normal vector along the normal direction
    // a little bit. If the translated point is within the volume, the normal
    // vector is inward, which has the orientation -1. If the point is outside
    // the volume, the normal vector is outward, which has the orientation +1.
    add_vector(root_point, normal, eps);

    return is_point_in_volume(volume_tag, root_point) ? -1 : 1;
  }


  template <int dim, int spacedim>
  void
  GmshManip<dim, spacedim>::get_oriented_volume_boundaries(
    const EntityTag         volume_tag,
    std::vector<EntityTag> &oriented_surface_tags,
    const double            eps)
  {
    gmsh::vectorpair volume_dimtag(1);
    volume_dimtag[0].first  = dim + 1;
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


  template <int dim, int spacedim>
  void
  GmshManip<dim, spacedim>::get_oriented_volume_boundaries(
    const EntityTag                                volume_tag,
    std::vector<EntityTag>                        &oriented_surface_tags,
    std::map<EntityTag, std::array<EntityTag, 2>> &face_to_subdomain,
    const double                                   eps)
  {
    gmsh::vectorpair volume_dimtag(1);
    volume_dimtag[0].first  = dim + 1;
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


#endif /* INCLUDE_ELECTRIC_FIELD_GMSH_MANIPULATION_H_ */
