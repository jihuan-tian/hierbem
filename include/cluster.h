/**
 * \file cluster.h
 * \brief Implementation of the class Cluster.
 * \ingroup hierarchical_matrices
 * \date 2021-04-18
 * \author Jihuan Tian
 */

#ifndef INCLUDE_CLUSTER_H_
#define INCLUDE_CLUSTER_H_

/**
 * \ingroup hierarchical_matrices
 * @{
 */

#include <deal.II/base/types.h>

#include <algorithm>
#include <iterator>
#include <vector>

#include "simple_bounding_box.h"

/**
 * Calculate the average cell sizes associated with those DoFs handled by the
 * given DoF handler object.
 *
 * The value doubled is used as an estimate for the diameter of the support set
 * of each DoF.
 *
 * @param dof_average_cell_size The returned list of average cell sizes. The
 * memory for this vector should be preallocated and initialized to zero before
 * calling this function.
 */
template <int dim, int spacedim, typename Number = double>
void
map_dofs_to_average_cell_size(const DoFHandler<dim, spacedim> &dof_handler,
                              std::vector<Number> &dof_average_cell_size)
{
  const unsigned int n_dofs = dof_handler.n_dofs();
  Assert(n_dofs == dof_average_cell_size.size(),
         ExcDimensionMismatch(n_dofs, dof_average_cell_size.size()));

  /**
   * Create the vector which stores the number of cells that share a common
   * DoF for each DoF.
   */
  std::vector<unsigned int> number_of_cells_sharing_dof(n_dofs, 0);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      /**
       * Get the diameter of the current cell.
       */
      Number cell_diameter = cell->diameter();

      /**
       * Get DoF indices local to this cell.
       */
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      for (auto index : dof_indices)
        {
          number_of_cells_sharing_dof.at(index) += 1;
          dof_average_cell_size.at(index) += cell_diameter;
        }
    }

  for (unsigned int i = 0; i < n_dofs; i++)
    {
      dof_average_cell_size.at(i) /= number_of_cells_sharing_dof.at(i);
    }
}

/**
 * Calculate the maximum cell sizes associated with those DoFs handled by the
 * given DoF handler object.
 *
 * The value doubled is used as an estimate for the diameter of the support
 * set of each DoF.
 *
 * @param dof_max_cell_size The returned list of maximum cell sizes. The memory
 * for this vector should be preallocated and initialized to zero before calling
 * this function.
 */
template <typename DoFHandlerType, typename Number = double>
void
map_dofs_to_max_cell_size(const DoFHandlerType &dof_handler,
                          std::vector<Number> & dof_max_cell_size)
{
  const unsigned int n_dofs = dof_handler.n_dofs();
  Assert(n_dofs == dof_max_cell_size.size(),
         ExcDimensionMismatch(n_dofs, dof_max_cell_size.size()));

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      /**
       * Get the diameter of the current cell.
       */
      Number cell_diameter = cell->diameter();

      /**
       * Get DoF indices local to this cell.
       */
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      for (auto index : dof_indices)
        {
          if (cell_diameter > dof_max_cell_size.at(index))
            {
              dof_max_cell_size.at(index) = cell_diameter;
            }
        }
    }
}

/**
 * Calculate the minimum cell sizes associated with those DoFs handled by the
 * given DoF handler object.
 *
 * The value doubled is used as an estimate for the diameter of the support set
 * of each DoF.
 *
 * @param dof_min_cell_size The returned list of average cell sizes. The memory
 * for this vector should be preallocated and initialized to zero before calling
 * this function.
 */
template <typename DoFHandlerType, typename Number = double>
void
map_dofs_to_min_cell_size(const DoFHandlerType &dof_handler,
                          std::vector<Number> & dof_min_cell_size)
{
  const unsigned int n_dofs = dof_handler.n_dofs();
  Assert(n_dofs == dof_min_cell_size.size(),
         ExcDimensionMismatch(n_dofs, dof_min_cell_size.size()));

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      /**
       * Get the diameter of the current cell.
       */
      Number cell_diameter = cell->diameter();

      /**
       * Get DoF indices local to this cell.
       */
      std::vector<types::global_dof_index> dof_indices(
        cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      for (auto index : dof_indices)
        {
          if (cell_diameter < dof_min_cell_size.at(index) ||
              dof_min_cell_size.at(index) == 0)
            {
              dof_min_cell_size.at(index) = cell_diameter;
            }
        }
    }
}

/**
 * \brief Class for an index cluster.
 *
 * The \p Cluster class contains both the DoF index set \p index_set and the
 * corresponding bounding box \p bbox.
 */
template <int spacedim, typename Number = double>
class Cluster
{
public:
  template <int spacedim1, typename Number1>
  friend std::ostream &
  operator<<(std::ostream &out, const Cluster<spacedim1, Number1> &cluster);

  template <int spacedim1, typename Number1>
  friend Number1
  calc_cluster_distance(
    const Cluster<spacedim1, Number1> &           cluster1,
    const Cluster<spacedim1, Number1> &           cluster2,
    const std::vector<Point<spacedim1, Number1>> &all_support_points);

  template <int spacedim1, typename Number1>
  friend Number1
  calc_cluster_distance(
    const Cluster<spacedim1, Number1> &           cluster1,
    const Cluster<spacedim1, Number1> &           cluster2,
    const std::vector<Point<spacedim1, Number1>> &all_support_points,
    const std::vector<Number1> &                  cell_size_at_dofs);

  /**
   * Check the equality of two clusters by comparing their index sets.
   *
   * This function firstly check the equality of the sizes/cardinalities of the
   * index sets in the two clusters. If their sizes are equal, then check the
   * contents.
   *
   * @param cluster1
   * @param cluster2
   * @return
   */
  template <int spacedim1, typename Number1>
  friend bool
  operator==(const Cluster<spacedim1, Number1> &cluster1,
             const Cluster<spacedim1, Number1> &cluster2);

  /**
   * Default constructor.
   */
  Cluster();

  /**
   * Constructor from an index set only without support points and associated
   * bounding box.
   * @param index_set
   */
  Cluster(const std::vector<types::global_dof_index> &index_set);

  /**
   * Constructor from an index set without cluster diameter correction.
   *
   * The bounding box will be recalculated.
   * @param index_set
   * @param all_support_points
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const std::vector<Point<spacedim, Number>> &all_support_points);

  /**
   * Constructor from an index set with cluster diameter correction.
   *
   * The bounding box will be recalculated.
   * @param index_set
   * @param all_support_points
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const std::vector<Point<spacedim, Number>> &all_support_points,
          const std::vector<Number> &                 cell_size_at_dofs);

  /**
   * Constructor from an index set and a bounding box without cluster diameter
   * correction.
   *
   * The input bounding box will be copied into the cluster without
   * recalculation. However, the diameter of the cluster is recalculated.
   * @param index_set
   * @param bbox
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const SimpleBoundingBox<spacedim, Number> & bbox,
          const std::vector<Point<spacedim, Number>> &all_support_points);

  /**
   * Constructor from an index set and a bounding box with cluster diameter
   * correction.
   *
   * The input bounding box will be copied into the cluster without
   * recalculation. However, the diameter of the cluster is recalculated.
   * @param index_set
   * @param bbox
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const SimpleBoundingBox<spacedim, Number> & bbox,
          const std::vector<Point<spacedim, Number>> &all_support_points,
          const std::vector<Number> &                 cell_size_at_dofs);

  /**
   * Copy constructor.
   */
  Cluster(const Cluster<spacedim, Number> &cluster);

  /**
   * Get the reference to the index set.
   */
  std::vector<types::global_dof_index> &
  get_index_set();

  /**
   * Get the reference to the index set (const version).
   */
  const std::vector<types::global_dof_index> &
  get_index_set() const;

  /**
   * Get the reference to the bounding box.
   */
  SimpleBoundingBox<spacedim, Number> &
  get_bounding_box();

  /**
   * Get the reference to the bounding box (const version).
   */
  const SimpleBoundingBox<spacedim, Number> &
  get_bounding_box() const;

  /**
   * Get the diameter of the cluster.
   */
  Number
  get_diameter() const;

  /**
   * Calculate the diameter of the cluster. There is no cell size correction.
   */
  Number
  calc_diameter(
    const std::vector<Point<spacedim, Number>> &all_support_points) const;

  /**
   * Calculate the diameter of the cluster. Cell size correction is applied.
   *
   * N.B. Doubled estimated cell size is adopted as an approximation of the
   * support set diameter \f${\rm diam}(Q_j)\f$. The correction is
   * calculated according to the following formula. \f[ \widetilde{\rm
   * diam}(\tau) := {\rm diam}(\hat{Q}_{\tau}) + \max_{j \in \tau}
   * {\rm diam}(Q_j) \f]
   * @param all_support_points
   * @param cell_size_at_dofs
   * @return
   */
  Number
  calc_diameter(const std::vector<Point<spacedim, Number>> &all_support_points,
                const std::vector<Number> &cell_size_at_dofs) const;

  /**
   * Calculate the minimum distance of the current cluster to the given cluster.
   * There is no cell size correction.
   */
  Number
  distance_to_cluster(
    const Cluster &                             cluster,
    const std::vector<Point<spacedim, Number>> &all_support_points) const;

  /**
   * Calculate the minimum distance of the current cluster to the given cluster.
   * Cell size correction is applied.
   */
  Number
  distance_to_cluster(
    const Cluster &                             cluster,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number> &                 cell_size_at_dofs) const;

  /**
   * Check if the index set of the current cluster is a subset of that of the
   * given cluster.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.</dd>
   * </dl>
   * @param cluster
   * @return
   */
  bool
  is_subset(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a proper subset of that of
   * the given cluster.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.</dd>
   * </dl>
   * @param cluster
   * @return
   */
  bool
  is_proper_subset(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a superset of that of the
   * given cluster.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.</dd>
   * </dl>
   * @param cluster
   * @return
   */
  bool
  is_superset(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a proper superset of that
   * of the given cluster.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.</dd>
   * </dl>
   * @param cluster
   * @return
   */
  bool
  is_proper_superset(const Cluster &cluster) const;

  /**
   * Calculate the intersection of the index sets of the current and the given
   * clusters.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.</dd>
   * </dl>
   * @param cluster
   * @param index_set_intersection
   */
  void
  intersect(const Cluster &                       cluster,
            std::vector<types::global_dof_index> &index_set_intersection) const;

  /**
   * Determine if the index set of the current cluster has a nonempty
   * intersection with the index set of the given cluster.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.</dd>
   * </dl>
   * @param cluster
   * @return
   */
  bool
  has_intersection(const Cluster &cluster) const;

  /**
   * Get the cardinality of the index set.
   * @return
   */
  std::size_t
  get_cardinality() const;

  /**
   * Determine if the cluster is large enough.
   *
   * @param n_min The size threshold value for determining if a cluster is large.
   * @return
   */
  bool
  is_large(unsigned int n_min) const;

private:
  std::vector<types::global_dof_index> index_set;
  SimpleBoundingBox<spacedim, Number>  bbox;
  Number                               diameter;
};


/**
 * Print out the cluster data.
 * @param out
 * @param cluster
 * @return
 */
template <int spacedim, typename Number>
std::ostream &
operator<<(std::ostream &out, const Cluster<spacedim, Number> &cluster)
{
  out << "Index set size: " << cluster.get_cardinality() << "\n";
  out << "Index set: [";
  for (auto index : cluster.index_set)
    {
      out << index << " ";
    }
  out << "]\n";
  out << "Bounding box: " << cluster.bbox;
  out << "Diameter: " << cluster.diameter;

  return out;
}


/**
 * Calculate the minimum distance between two clusters. This calculation has
 * no mesh size correction.
 *
 * The calculation is based on measuring the distance between each pair of
 * support points contained in the clusters, which prevents the distance
 * calculation between two support sets.
 * @param cluster1
 * @param cluster2
 * @param all_support_points A list of support point coordinates which are ordered by DoF indices.
 * @return
 */
template <int spacedim, typename Number = double>
Number
calc_cluster_distance(
  const Cluster<spacedim, Number> &           cluster1,
  const Cluster<spacedim, Number> &           cluster2,
  const std::vector<Point<spacedim, Number>> &all_support_points)
{
  // Calculate the total number of point pairs.
  const unsigned int point_pair_num =
    cluster1.get_index_set().size() * cluster2.get_index_set().size();

  // Create a linearized list storing all the distance values.
  std::vector<Number> point_pair_distance(point_pair_num);

  unsigned int counter = 0;
  for (const auto &index1 : cluster1.get_index_set())
    {
      for (const auto &index2 : cluster2.get_index_set())
        {
          point_pair_distance.at(counter) =
            all_support_points.at(index1).distance(
              all_support_points.at(index2));

          counter++;
        }
    }

  return (*std::min_element(point_pair_distance.cbegin(),
                            point_pair_distance.cend()));
}


/**
 * Calculate the minimum distance between two clusters. This calculation has
 * the mesh size correction.
 *
 * The calculation is based on measuring the distance between each pair of
 * support points contained in the clusters, which prevents the distance
 * calculation between two support sets.
 * @param cluster1
 * @param cluster2
 * @param all_support_points A list of support point coordinates which are ordered by DoF indices.
 * @param cell_size_at_dofs The list of estimated cell size values at DoF support points.
 * @return
 */
template <int spacedim, typename Number = double>
Number
calc_cluster_distance(
  const Cluster<spacedim, Number> &           cluster1,
  const Cluster<spacedim, Number> &           cluster2,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs)
{
  /**
   * Calculate the uncorrected cluster distance.
   */
  Number uncorrected_cluster_distance =
    calc_cluster_distance(cluster1, cluster2, all_support_points);

  /**
   * Get the maximum diameter of the support sets for different DoFs, which is 2
   * times of the cell size associated with the corresponding DoF.
   */
  Number max_dof_cell_size = 0.0;

  for (const auto &index : cluster1.get_index_set())
    {
      if (cell_size_at_dofs.at(index) > max_dof_cell_size)
        {
          max_dof_cell_size = cell_size_at_dofs.at(index);
        }
    }

  for (const auto &index : cluster2.get_index_set())
    {
      if (cell_size_at_dofs.at(index) > max_dof_cell_size)
        {
          max_dof_cell_size = cell_size_at_dofs.at(index);
        }
    }

  Number distance_correction = max_dof_cell_size * 2;

  /**
   * Ensure the positivity of the returned cluster distance.
   */
  if (uncorrected_cluster_distance > distance_correction)
    {
      return uncorrected_cluster_distance - distance_correction;
    }
  else
    {
      return uncorrected_cluster_distance;
    }
}


template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster()
  : index_set(0)
  , bbox()
  , diameter(0)
{}


template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set)
  : index_set(index_set)
  , bbox()
  , diameter(0)
{}


template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const std::vector<Point<spacedim, Number>> &all_support_points)
  : index_set(index_set)
  , bbox(index_set, all_support_points)
  , diameter(calc_diameter(all_support_points))
{}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs)
  : index_set(index_set)
  , bbox(index_set, all_support_points)
  , diameter(calc_diameter(all_support_points, cell_size_at_dofs))
{}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const SimpleBoundingBox<spacedim, Number> & bbox,
  const std::vector<Point<spacedim, Number>> &all_support_points)
  : index_set(index_set)
  , bbox(bbox)
  , diameter(calc_diameter(all_support_points))
{}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const SimpleBoundingBox<spacedim, Number> & bbox,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs)
  : index_set(index_set)
  , bbox(bbox)
  , diameter(calc_diameter(all_support_points, cell_size_at_dofs))
{}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(const Cluster<spacedim, Number> &cluster)
  : index_set(cluster.index_set)
  , bbox(cluster.bbox)
  , diameter(cluster.diameter)
{}

template <int spacedim, typename Number>
std::vector<types::global_dof_index> &
Cluster<spacedim, Number>::get_index_set()
{
  return index_set;
}

template <int spacedim, typename Number>
const std::vector<types::global_dof_index> &
Cluster<spacedim, Number>::get_index_set() const
{
  return index_set;
}

template <int spacedim, typename Number>
SimpleBoundingBox<spacedim, Number> &
Cluster<spacedim, Number>::get_bounding_box()
{
  return bbox;
}

template <int spacedim, typename Number>
const SimpleBoundingBox<spacedim, Number> &
Cluster<spacedim, Number>::get_bounding_box() const
{
  return bbox;
}

template <int spacedim, typename Number>
Number
Cluster<spacedim, Number>::get_diameter() const
{
  return diameter;
}

template <int spacedim, typename Number>
Number
Cluster<spacedim, Number>::calc_diameter(
  const std::vector<Point<spacedim, Number>> &all_support_points) const
{
  // Number of support points in the cluster.
  const unsigned int n = index_set.size();

  /**
   * Calculate the number of point pairs in the cluster. Let \f$[0, 1, 2, 3, 4,
   * 5]\f$ be the indices of support points in the cluster, whose pairwise
   * inter-distance will be calculated. The calculation is only needed for the
   * marked pairs of points as shown below.
   *
   * \code
   *   0 1 2 3 4 5
   * 0   - - - - -
   * 1     - - - -
   * 2       - - -
   * 3         - -
   * 4           -
   * 5
   * \endcode
   *
   * Therefore, the total number of effective point pairs is \f$\frac{n^2 -
   * n}{2}\f$.
   */
  const unsigned int point_pair_num = n * (n - 1) / 2;

  std::vector<Number> point_pair_distance(point_pair_num);

  unsigned int counter = 0;
  for (unsigned int i = 0; i < (n - 1); i++)
    {
      for (unsigned int j = i + 1; j < n; j++)
        {
          point_pair_distance.at(counter) =
            all_support_points.at(index_set.at(i))
              .distance(all_support_points.at(index_set.at(j)));

          counter++;
        }
    }

  if (n > 1)
    {
      return (*std::max_element(point_pair_distance.cbegin(),
                                point_pair_distance.cend()));
    }
  else
    {
      return 0;
    }
}

template <int spacedim, typename Number>
Number
Cluster<spacedim, Number>::calc_diameter(
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs) const
{
  Number uncorrected_diameter = calc_diameter(all_support_points);

  Number max_dof_cell_size = 0;

  for (const auto &index : index_set)
    {
      if (cell_size_at_dofs.at(index) > max_dof_cell_size)
        {
          max_dof_cell_size = cell_size_at_dofs.at(index);
        }
    }

  return uncorrected_diameter + 2 * max_dof_cell_size;
}

template <int spacedim, typename Number>
Number
Cluster<spacedim, Number>::distance_to_cluster(
  const Cluster &                             cluster,
  const std::vector<Point<spacedim, Number>> &all_support_points) const
{
  return calc_cluster_distance((*this), cluster, all_support_points);
}

template <int spacedim, typename Number>
Number
Cluster<spacedim, Number>::distance_to_cluster(
  const Cluster &                             cluster,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs) const
{
  return calc_cluster_distance((*this),
                               cluster,
                               all_support_points,
                               cell_size_at_dofs);
}


template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::is_subset(const Cluster &cluster) const
{
  return (std::includes(cluster.index_set.begin(),
                        cluster.index_set.end(),
                        this->index_set.begin(),
                        this->index_set.end()));
}


template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::is_proper_subset(const Cluster &cluster) const
{
  if (std::includes(cluster.index_set.begin(),
                    cluster.index_set.end(),
                    this->index_set.begin(),
                    this->index_set.end()))
    {
      if (cluster.index_set.size() == this->index_set.size())
        {
          return false;
        }
      else
        {
          return true;
        }
    }
  else
    {
      return false;
    }
}


template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::is_superset(const Cluster &cluster) const
{
  return (std::includes(this->index_set.begin(),
                        this->index_set.end(),
                        cluster.index_set.begin(),
                        cluster.index_set.end()));
}


template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::is_proper_superset(const Cluster &cluster) const
{
  if (std::includes(this->index_set.begin(),
                    this->index_set.end(),
                    cluster.index_set.begin(),
                    cluster.index_set.end()))
    {
      if (cluster.index_set.size() == this->index_set.size())
        {
          return false;
        }
      else
        {
          return true;
        }
    }
  else
    {
      return false;
    }
}


template <int spacedim, typename Number>
void
Cluster<spacedim, Number>::intersect(
  const Cluster &                       cluster,
  std::vector<types::global_dof_index> &index_set_intersection) const
{
  index_set_intersection.clear();

  std::set_intersection(this->index_set.begin(),
                        this->index_set.end(),
                        cluster.index_set.begin(),
                        cluster.index_set.end(),
                        std::back_inserter(index_set_intersection));
}


template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::has_intersection(const Cluster &cluster) const
{
  std::vector<types::global_dof_index> index_set_intersection;
  this->intersect(cluster, index_set_intersection);

  if (index_set_intersection.size() > 0)
    {
      return true;
    }
  else
    {
      return false;
    }
}


template <int spacedim, typename Number>
std::size_t
Cluster<spacedim, Number>::get_cardinality() const
{
  return index_set.size();
}

template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::is_large(unsigned int n_min) const
{
  if (index_set.size() > n_min)
    {
      return true;
    }
  else
    {
      return false;
    }
}


template <int spacedim, typename Number>
bool
operator==(const Cluster<spacedim, Number> &cluster1,
           const Cluster<spacedim, Number> &cluster2)
{
  if (cluster1.index_set.size() == cluster2.index_set.size())
    {
      return (cluster1.index_set == cluster2.index_set);
    }
  else
    {
      return false;
    }
}

/**
 * @}
 */

#endif /* INCLUDE_CLUSTER_H_ */
