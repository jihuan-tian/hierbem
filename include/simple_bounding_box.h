#ifndef SIMPLE_BOUNDING_BOX_H
#define SIMPLE_BOUNDING_BOX_H

#include <deal.II/base/point.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping.h>

#include <utility> // in which std::pair is defined.
#include <vector>

using namespace dealii;

namespace IdeoBEM
{
  /**
   * Class implementing a simple axis-parallel bounding box. N.B. This class
   * only
   * has the \p spacedim template argument but without the \p dim template
   * argument, which means an axis-parallel bounding box always has the same
   * dimension as the specified space dimension. Therefore, the bounding box for
   * a 2D surface in 3D space is still a 3D box.
   */
  template <int spacedim, typename Number = double>
  class SimpleBoundingBox
  {
  public:
    template <int spacedim1, typename Number1>
    friend std::ostream &
    operator<<(std::ostream                                &out,
               const SimpleBoundingBox<spacedim1, Number1> &bbox);

    /**
     * Default constructor with two boundary points being both zeros.
     */
    SimpleBoundingBox();

    /**
     * Constructor from two corner points.
     */
    SimpleBoundingBox(const Point<spacedim, Number> &corner1,
                      const Point<spacedim, Number> &corner2);


    /**
     * Constructor from a pair of corner points.
     * @param point_pair
     */
    SimpleBoundingBox(const std::pair<Point<spacedim, Number>,
                                      Point<spacedim, Number>> &point_pair);


    /**
     * Constructor from a vector of points.
     */
    SimpleBoundingBox(const std::vector<Point<spacedim, Number>> &points);


    /**
     * Constructor from a \p Mapping and a \p DoFHandler. N.B. This is a template
     * constructor with the argument dim as manifold dimension. The resulted
     * bounding box contains the support points associated with all DoFs within
     * the DoFHandler.
     */
    template <int dim>
    SimpleBoundingBox(const Mapping<dim, spacedim>    &mapping,
                      const DoFHandler<dim, spacedim> &dof_handler);


    /**
     * Constructor from a specified DoF index set. The support point coordinates
     * for all DoFs have been placed into \p all_support_points.
     * @param dof_indices The DoF index set, for which the bounding box is to be generated.
     * @param all_support_points The const reference to the list of all support points associated with a DoFHandler.
     */
    SimpleBoundingBox(
      const std::vector<types::global_dof_index> &dof_indices,
      const std::vector<Point<spacedim, Number>> &all_support_points);


    /**
     * Copy constructor
     */
    SimpleBoundingBox(const SimpleBoundingBox<spacedim, Number> &bbox);

    /**
     * Calculate the volume of the bounding box.
     * @return Volume of the bounding box.
     */
    Number
    volume() const;

    /**
     * Get the two corner points of the bounding box (mutable version).
     * @return The reference to the pair of corner points. The point with a smaller coordinate in each dimension is the first.
     */
    std::pair<Point<spacedim, Number>, Point<spacedim, Number>> &
    get_boundary_points();

    /**
     * Get the two corner points of the bounding box (const version).
     * @return The const reference to the pair of corner points. The point with a smaller coordinate in each dimension is the first.
     */
    const std::pair<Point<spacedim, Number>, Point<spacedim, Number>> &
    get_boundary_points() const;

    /**
     * Determine if a given point lies within the bounding box.
     */
    bool
    point_inside(const Point<spacedim, Number> &p) const;

    /**
     * Get the index to the coordinate component, which has the longest
     * dimension.
     * @return Index to the coordinate component, which should be in the range \f$[0, {\rm spacedim})\f$.
     */
    unsigned int
    coordinate_index_with_longest_dimension() const;

    /**
     * Bisect the bounding box along the longest coordinate direction.
     * @return
     */
    std::pair<SimpleBoundingBox<spacedim, Number>,
              SimpleBoundingBox<spacedim, Number>>
    divide_geometrically() const;

  private:
    /**
     * Calculate the bounding box from a list of points.
     */
    void
    calculate_bounding_box(const std::vector<Point<spacedim, Number>> &points);

    /**
     * Calculate the bounding box from a list of DoF indices.
     * @param dof_indices The DoF index set.
     * @param all_support_points The const reference to the list of all support points associated with a DoFHandler.
     */
    void
    calculate_bounding_box(
      const std::vector<types::global_dof_index> &dof_indices,
      const std::vector<Point<spacedim, Number>> &all_support_points);


    /**
     * Two corner points of the bounding box. The point in the pair is the
     * bottom corner, i.e. it has a smaller component coordinate in each
     * dimension; while the second point is the top corner.
     */
    std::pair<Point<spacedim, Number>, Point<spacedim, Number>> boundary_points;
  };

  template <int spacedim, typename Number>
  std::ostream &
  operator<<(std::ostream &out, const SimpleBoundingBox<spacedim, Number> &bbox)
  {
    out << "[" << bbox.boundary_points.first << "], ["
        << bbox.boundary_points.second << "]\n";

    return out;
  }

  template <int spacedim, typename Number>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox()
    : boundary_points(Point<spacedim, Number>(), Point<spacedim, Number>())
  {}

  template <int spacedim, typename Number>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox(
    const Point<spacedim, Number> &corner1,
    const Point<spacedim, Number> &corner2)
    : boundary_points(corner1, corner2)
  {}

  template <int spacedim, typename Number>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox(
    const std::pair<Point<spacedim, Number>, Point<spacedim, Number>>
      &point_pair)
    : boundary_points(point_pair)
  {}

  template <int spacedim, typename Number>
  void
  SimpleBoundingBox<spacedim, Number>::calculate_bounding_box(
    const std::vector<Point<spacedim, Number>> &points)
  {
    /**
     * Initialize the two boundary points to be the first point in the list.
     */
    boundary_points.first  = points.at(0);
    boundary_points.second = points.at(0);

    /**
     * Calculate the minimum and the maximum coordinate for each dimension.
     */
    for (const auto &point : points)
      {
        for (unsigned int d = 0; d < spacedim; d++)
          {
            if (point(d) < boundary_points.first(d))
              {
                boundary_points.first(d) = point(d);
              }
            else if (point(d) > boundary_points.second(d))
              {
                boundary_points.second(d) = point(d);
              }
          }
      }
  }

  template <int spacedim, typename Number>
  void
  SimpleBoundingBox<spacedim, Number>::calculate_bounding_box(
    const std::vector<types::global_dof_index> &dof_indices,
    const std::vector<Point<spacedim, Number>> &all_support_points)
  {
    /**
     * Initialize the two boundary points to be the point associated with the
     * first DoF index.
     */
    boundary_points.first  = all_support_points.at(dof_indices.at(0));
    boundary_points.second = boundary_points.first;

    /**
     * Calculate the minimum and the maximum coordinate for each dimension.
     */
    for (const auto dof_index : dof_indices)
      {
        const Point<spacedim, Number> &point = all_support_points.at(dof_index);
        for (unsigned int d = 0; d < spacedim; d++)
          {
            if (point(d) < boundary_points.first(d))
              {
                boundary_points.first(d) = point(d);
              }
            else if (point(d) > boundary_points.second(d))
              {
                boundary_points.second(d) = point(d);
              }
          }
      }
  }

  template <int spacedim, typename Number>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox(
    const std::vector<Point<spacedim, Number>> &points)
  {
    calculate_bounding_box(points);
  }

  template <int spacedim, typename Number>
  template <int dim>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox(
    const Mapping<dim, spacedim>    &mapping,
    const DoFHandler<dim, spacedim> &dof_handler)
  {
    // Allocate memory for the vector of support points, which are associated
    // with all the DoFs in the DoFHandler.
    std::vector<Point<spacedim, Number>> support_points(dof_handler.n_dofs());
    // Extract the support points for all DoFs.
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
    // Calculate the bounding box for all the support points.
    calculate_bounding_box(support_points);
  }

  template <int spacedim, typename Number>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox(
    const std::vector<types::global_dof_index> &dof_indices,
    const std::vector<Point<spacedim, Number>> &all_support_points)
  {
    // Calculate the bounding box for the support points associated with the DoF
    // index set.
    calculate_bounding_box(dof_indices, all_support_points);
  }

  template <int spacedim, typename Number>
  SimpleBoundingBox<spacedim, Number>::SimpleBoundingBox(
    const SimpleBoundingBox<spacedim, Number> &bbox)
    : boundary_points(bbox.boundary_points)
  {}

  template <int spacedim, typename Number>
  Number
  SimpleBoundingBox<spacedim, Number>::volume() const
  {
    Number v = 1.0;

    for (unsigned int d = 0; d < spacedim; d++)
      {
        v *= (boundary_points.second(d) - boundary_points.first(d));
      }

    return v;
  }

  template <int spacedim, typename Number>
  unsigned int
  SimpleBoundingBox<spacedim, Number>::coordinate_index_with_longest_dimension()
    const
  {
    unsigned int coordinate_index  = -1;
    Number       longest_dimension = 0.0;
    for (unsigned int d = 0; d < spacedim; d++)
      {
        Number current_dimension =
          boundary_points.second(d) - boundary_points.first(d);

        if (current_dimension > longest_dimension)
          {
            longest_dimension = current_dimension;
            coordinate_index  = d;
          }
      }

    AssertIndexRange(coordinate_index, spacedim);

    return coordinate_index;
  }

  template <int spacedim, typename Number>
  std::pair<SimpleBoundingBox<spacedim, Number>,
            SimpleBoundingBox<spacedim, Number>>
  SimpleBoundingBox<spacedim, Number>::divide_geometrically() const
  {
    /**
     * Calculate the coordinate index which has the longest dimension.
     */
    unsigned int coordinate_index = coordinate_index_with_longest_dimension();

    SimpleBoundingBox<spacedim, Number> first_bbox(*this);
    SimpleBoundingBox<spacedim, Number> second_bbox(*this);

    /**
     * Modify the \p coordinate_index 'th coordinate of the top corner point in
     * the first box.
     */
    std::pair<Point<spacedim, Number>, Point<spacedim, Number>>
      &first_bbox_boundary_points = first_bbox.get_boundary_points();
    first_bbox_boundary_points.second(coordinate_index) =
      first_bbox_boundary_points.first(coordinate_index) +
      0.5 * (first_bbox_boundary_points.second(coordinate_index) -
             first_bbox_boundary_points.first(coordinate_index));

    /**
     * Modify the \p coordinate_index 'th coordinate of the bottom corner point in
     * the second box.
     */
    std::pair<Point<spacedim, Number>, Point<spacedim, Number>>
      &second_bbox_boundary_points = second_bbox.get_boundary_points();
    second_bbox_boundary_points.first(coordinate_index) =
      first_bbox_boundary_points.second(coordinate_index);


    return std::pair<SimpleBoundingBox<spacedim, Number>,
                     SimpleBoundingBox<spacedim, Number>>(first_bbox,
                                                          second_bbox);
  }

  template <int spacedim, typename Number>
  std::pair<Point<spacedim, Number>, Point<spacedim, Number>> &
  SimpleBoundingBox<spacedim, Number>::get_boundary_points()
  {
    return boundary_points;
  }

  template <int spacedim, typename Number>
  const std::pair<Point<spacedim, Number>, Point<spacedim, Number>> &
  SimpleBoundingBox<spacedim, Number>::get_boundary_points() const
  {
    return boundary_points;
  }

  template <int spacedim, typename Number>
  bool
  SimpleBoundingBox<spacedim, Number>::point_inside(
    const Point<spacedim, Number> &p) const
  {
    bool is_inside = true;

    for (unsigned int d = 0; d < spacedim; d++)
      {
        /**
         * Make a predicate on if the point lies outside the bounding box.
         */
        if (p(d) < boundary_points.first(d) || p(d) > boundary_points.second(d))
          {
            is_inside = false;

            break;
          }
      }

    return is_inside;
  }
} // namespace IdeoBEM

#endif
