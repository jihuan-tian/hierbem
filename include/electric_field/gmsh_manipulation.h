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

namespace HierBEM
{
  using EntityTag = int;

  class GmshManip
  {
  public:
    /**
     * Check if a point with the given Cartesian coordinates is inside a volume.
     * @pre
     * @post
     * @param volume_tag
     * @param coord 3D Cartesian coordinates
     * @return
     */
    static bool
    is_point_in_volume(const EntityTag            volume_tag,
                       const std::vector<double> &coord);

    /**
     * Get the normal vector of a cell (dim=2) on the specified surface.
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
     * volume. The former is determined by the local coordinate chart
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
     * Compute \f$x = x + \alpha y\f$, where both \f$x\f$ and \f$y\f$ are
     * @p std::vector.
     *
     * @pre
     * @post
     * @param x
     * @param y
     * @param alpha
     */
    static void
    add_vector(std::vector<double>       &x,
               const std::vector<double> &y,
               double                     alpha);

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

    static void
    get_oriented_volume_boundaries(
      const EntityTag                                volume_tag,
      std::vector<EntityTag>                        &oriented_surface_tags,
      std::map<EntityTag, std::array<EntityTag, 2>> &face_to_subdomain,
      const double                                   eps = 1e-5);
  };
} // namespace HierBEM


#endif /* INCLUDE_ELECTRIC_FIELD_GMSH_MANIPULATION_H_ */
