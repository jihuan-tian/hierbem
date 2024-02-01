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
#include <array>
#include <iterator>
#include <vector>

#include "generic_functors.h"
#include "simple_bounding_box.h"

namespace HierBEM
{
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
      const Cluster<spacedim1, Number1>            &cluster1,
      const Cluster<spacedim1, Number1>            &cluster2,
      const std::vector<Point<spacedim1, Number1>> &all_support_points);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1> &cluster1,
      const Cluster<spacedim1, Number1> &cluster2,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim1, Number1>> &all_support_points);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1>            &cluster1,
      const Cluster<spacedim1, Number1>            &cluster2,
      const std::vector<Point<spacedim1, Number1>> &all_support_points1,
      const std::vector<Point<spacedim1, Number1>> &all_support_points2);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1> &cluster1,
      const Cluster<spacedim1, Number1> &cluster2,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering1,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering2,
      const std::vector<Point<spacedim1, Number1>> &all_support_points1,
      const std::vector<Point<spacedim1, Number1>> &all_support_points2);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1>            &cluster1,
      const Cluster<spacedim1, Number1>            &cluster2,
      const std::vector<Point<spacedim1, Number1>> &all_support_points,
      const std::vector<Number1>                   &cell_size_at_dofs);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1> &cluster1,
      const Cluster<spacedim1, Number1> &cluster2,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim1, Number1>> &all_support_points,
      const std::vector<Number1>                   &cell_size_at_dofs);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1>            &cluster1,
      const Cluster<spacedim1, Number1>            &cluster2,
      const std::vector<Point<spacedim1, Number1>> &all_support_points1,
      const std::vector<Point<spacedim1, Number1>> &all_support_points2,
      const std::vector<Number1>                   &cell_size_at_dofs1,
      const std::vector<Number1>                   &cell_size_at_dofs2);

    template <int spacedim1, typename Number1>
    friend Number1
    calc_cluster_distance(
      const Cluster<spacedim1, Number1> &cluster1,
      const Cluster<spacedim1, Number1> &cluster2,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering1,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering2,
      const std::vector<Point<spacedim1, Number1>> &all_support_points1,
      const std::vector<Point<spacedim1, Number1>> &all_support_points2,
      const std::vector<Number1>                   &cell_size_at_dofs1,
      const std::vector<Number1>                   &cell_size_at_dofs2);

    /**
     * Check the equality of two clusters by comparing their index sets.
     *
     * This function firstly check the equality of the sizes/cardinalities of
     * the index sets in the two clusters. If their sizes are equal, then
     * 1. if the size is larger than 0, which means the index set is not empty
     * and the index range is still not generated, then compare the contents in
     * the index set.
     * 2. if the size is 0, which means the index set has been cleared and the
     * index range has been generated, then compare the contents in the index
     * range.
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
            const std::vector<Number>                  &cell_size_at_dofs);

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
            const SimpleBoundingBox<spacedim, Number>  &bbox,
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
            const SimpleBoundingBox<spacedim, Number>  &bbox,
            const std::vector<Point<spacedim, Number>> &all_support_points,
            const std::vector<Number>                  &cell_size_at_dofs);

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
     * Get the reference to the index range, which is in the internal numbering.
     * @return
     */
    std::array<types::global_dof_index, 2> &
    get_index_range();

    /**
     * Get the reference to the index range, which is in the internal numbering
     * (const version).
     * @return
     */
    const std::array<types::global_dof_index, 2> &
    get_index_range() const;

    /**
     * Set the index range, which is in the internal numbering. After the index
     * range is set, the original index set will be immediately cleared for
     * saving memory.
     *
     * @param lower_bound
     * @param upper_bound
     */
    void
    set_index_range(const types::global_dof_index lower_bound,
                    const types::global_dof_index pass_upper_bound);

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
     * Calculate the diameter of the cluster. There is no cell size correction.
     *
     * \mynote{In this version, the index set held by the cluster is empty and
     * the index range takes effect.}
     *
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @return
     */
    Number
    calc_diameter(
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points) const;

    /**
     * Calculate the diameter of the cluster. Cell size correction is applied.
     *
     * N.B. Doubled estimated cell size is adopted as an approximation of the
     * support set diameter \f${\rm diam}(Q_j)\f$. The correction is
     * calculated according to the following formula. \f[ \widetilde{\rm
     * diam}(\tau) := {\rm diam}(\hat{Q}_{\tau}) + \max_{j \in \tau}
     * {\rm diam}(Q_j) \f]
     *
     * @param all_support_points
     * @param cell_size_at_dofs
     * @return
     */
    Number
    calc_diameter(
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs) const;

    /**
     * Calculate the diameter of the cluster. Cell size correction is applied.
     *
     * N.B. Doubled estimated cell size is adopted as an approximation of the
     * support set diameter \f${\rm diam}(Q_j)\f$. The correction is
     * calculated according to the following formula. \f[ \widetilde{\rm
     * diam}(\tau) := {\rm diam}(\hat{Q}_{\tau}) + \max_{j \in \tau}
     * {\rm diam}(Q_j) \f]
     *
     * \mynote{In this version, the index set held by the cluster is empty and
     * the index range takes effect.}
     *
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @param cell_size_at_dofs
     * @return
     */
    Number
    calc_diameter(
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. There is no cell size correction.
     *
     * \mynote{The index sets held by the two clusters share a same external
     * DoF numbering.}
     */
    Number
    distance_to_cluster(
      const Cluster                              &cluster,
      const std::vector<Point<spacedim, Number>> &all_support_points) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. There is no cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters share a same internal DoF numbering, which need to be mapped to
     * the external numbering for accessing the list of support point
     * coordinates.}
     *
     * @param cluster
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @return
     */
    Number
    distance_to_cluster(
      const Cluster &cluster,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. There is no cell size correction.
     *
     * \mynote{The index sets held by the two clusters refer to two different
     * external DoF numberings.}
     *
     * @param cluster
     * @param all_support_points1
     * @param all_support_points2
     * @return
     */
    Number
    distance_to_cluster(
      const Cluster                              &cluster,
      const std::vector<Point<spacedim, Number>> &all_support_points1,
      const std::vector<Point<spacedim, Number>> &all_support_points2) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. There is no cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters refer to two different internal DoF numberings, which need to be
     * mapped to their corresponding external numberings for accessing the list
     * of support point coordinates.}
     *
     * @param cluster
     * @param internal_to_external_dof_numbering1
     * @param internal_to_external_dof_numbering2
     * @param all_support_points1
     * @param all_support_points2
     * @return
     */
    Number
    distance_to_cluster(
      const Cluster &cluster,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering1,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering2,
      const std::vector<Point<spacedim, Number>> &all_support_points1,
      const std::vector<Point<spacedim, Number>> &all_support_points2) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. Cell size correction is applied.
     *
     * \mynote{The index sets held by the two clusters share a same external
     * DoF numbering.}
     */
    Number
    distance_to_cluster(
      const Cluster                              &cluster,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. Cell size correction is applied.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters share a same internal DoF numbering, which need to be mapped to
     * the external numbering for accessing the list of support point
     * coordinates.}
     *
     * @param cluster
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @param cell_size_at_dofs
     * @return
     */
    Number
    distance_to_cluster(
      const Cluster &cluster,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. Cell size correction is applied.
     *
     * \mynote{The index sets held by the two clusters refer to two different
     * external DoF numberings.}
     *
     * @param cluster
     * @param all_support_points1
     * @param all_support_points2
     * @param cell_size_at_dofs1
     * @param cell_size_at_dofs2
     * @return
     */
    Number
    distance_to_cluster(
      const Cluster                              &cluster,
      const std::vector<Point<spacedim, Number>> &all_support_points1,
      const std::vector<Point<spacedim, Number>> &all_support_points2,
      const std::vector<Number>                  &cell_size_at_dofs1,
      const std::vector<Number>                  &cell_size_at_dofs2) const;

    /**
     * Calculate the minimum distance of the current cluster to the given
     * cluster. Cell size correction is applied.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters refer to two different internal DoF numberings, which need to be
     * mapped to their corresponding external numberings for accessing the list
     * of support point coordinates.}
     *
     * @param cluster
     * @param internal_to_external_dof_numbering1
     * @param internal_to_external_dof_numbering2
     * @param all_support_points1
     * @param all_support_points2
     * @param cell_size_at_dofs1
     * @param cell_size_at_dofs2
     * @return
     */
    Number
    distance_to_cluster(
      const Cluster &cluster,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering1,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering2,
      const std::vector<Point<spacedim, Number>> &all_support_points1,
      const std::vector<Point<spacedim, Number>> &all_support_points2,
      const std::vector<Number>                  &cell_size_at_dofs1,
      const std::vector<Number>                  &cell_size_at_dofs2) const;

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
     * Check if the index set of the current cluster is a proper subset of that
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
    is_proper_subset(const Cluster &cluster) const;

    /**
     * Check if the index set of the current cluster is a superset of that of
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
    is_superset(const Cluster &cluster) const;

    /**
     * Check if the index set of the current cluster is a proper superset of
     * that of the given cluster.
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
    intersect(
      const Cluster                        &cluster,
      std::vector<types::global_dof_index> &index_set_intersection) const;

    /**
     * Calculate the intersection of the index ranges of the current and the
     * given clusters.
     *
     * @param cluster
     * @param index_range_intersection
     */
    void
    intersect(
      const Cluster                          &cluster,
      std::array<types::global_dof_index, 2> &index_range_intersection) const;

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
    /**
     * The list of DoF indices.
     */
    std::vector<types::global_dof_index> index_set;
    /**
     * The DoF index range in the internal numbering.
     */
    std::array<types::global_dof_index, 2> index_range;
    SimpleBoundingBox<spacedim, Number>    bbox;
    Number                                 diameter;
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
    out << "Index set (external numbering): [";
    for (auto index : cluster.index_set)
      {
        out << index << " ";
      }
    out << "]\n";
    out << "Index range (internal numbering): [" << cluster.index_range[0]
        << " " << cluster.index_range[1] << ")\n";
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
   *
   * \mynote{The index sets held by the two clusters share a same external
   * DoF numbering.}
   *
   * @param cluster1
   * @param cluster2
   * @param all_support_points A list of support point coordinates which are
   * ordered by external DoF indices.
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number>            &cluster1,
    const Cluster<spacedim, Number>            &cluster2,
    const std::vector<Point<spacedim, Number>> &all_support_points)
  {
    Number cluster_distance =
      all_support_points.at(cluster1.get_index_set().at(0))
        .distance(all_support_points.at(cluster2.get_index_set().at(0)));
    Number point_pair_distance;

    for (const auto &index1 : cluster1.get_index_set())
      {
        for (const auto &index2 : cluster2.get_index_set())
          {
            point_pair_distance = all_support_points.at(index1).distance(
              all_support_points.at(index2));

            if (point_pair_distance < cluster_distance)
              {
                cluster_distance = point_pair_distance;
              }
          }
      }

    return cluster_distance;
  }


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * no mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters share a same internal DoF numbering, which need to be mapped to
   * the external numbering for accessing the list of support point
   * coordinates.}
   *
   * @param cluster1
   * @param cluster2
   * @param internal_to_external_dof_numbering
   * @param all_support_points
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number> &cluster1,
    const Cluster<spacedim, Number> &cluster2,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points)
  {
    const std::array<types::global_dof_index, 2> &cluster1_index_range =
      cluster1.get_index_range();
    const std::array<types::global_dof_index, 2> &cluster2_index_range =
      cluster2.get_index_range();

    Number cluster_distance =
      all_support_points
        .at(internal_to_external_dof_numbering[cluster1_index_range[0]])
        .distance(all_support_points.at(
          internal_to_external_dof_numbering[cluster2_index_range[0]]));
    Number point_pair_distance;

    for (types::global_dof_index index1 = cluster1_index_range[0];
         index1 < cluster1_index_range[1];
         index1++)
      {
        for (types::global_dof_index index2 = cluster2_index_range[0];
             index2 < cluster2_index_range[1];
             index2++)
          {
            point_pair_distance =
              all_support_points.at(internal_to_external_dof_numbering[index1])
                .distance(all_support_points.at(
                  internal_to_external_dof_numbering[index2]));

            if (point_pair_distance < cluster_distance)
              {
                cluster_distance = point_pair_distance;
              }
          }
      }

    return cluster_distance;
  }


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * no mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * \mynote{The index sets held by the two clusters refer to two different
   * external DoF numberings.}
   *
   * @param cluster1
   * @param cluster2
   * @param all_support_points1
   * @param all_support_points2
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number>            &cluster1,
    const Cluster<spacedim, Number>            &cluster2,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2)
  {
    Number cluster_distance =
      all_support_points1.at(cluster1.get_index_set().at(0))
        .distance(all_support_points2.at(cluster2.get_index_set().at(0)));
    Number point_pair_distance;

    for (const auto &index1 : cluster1.get_index_set())
      {
        for (const auto &index2 : cluster2.get_index_set())
          {
            point_pair_distance = all_support_points1.at(index1).distance(
              all_support_points2.at(index2));

            if (point_pair_distance < cluster_distance)
              {
                cluster_distance = point_pair_distance;
              }
          }
      }

    return cluster_distance;
  }


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * no mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters refer to two different internal DoF numberings, which need to be
   * mapped to their corresponding external numberings for accessing the list of
   * support point coordinates.}
   *
   * @param cluster1
   * @param cluster2
   * @param internal_to_external_dof_numbering1
   * @param internal_to_external_dof_numbering2
   * @param all_support_points1
   * @param all_support_points2
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number> &cluster1,
    const Cluster<spacedim, Number> &cluster2,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering1,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering2,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2)
  {
    const std::array<types::global_dof_index, 2> &cluster1_index_range =
      cluster1.get_index_range();
    const std::array<types::global_dof_index, 2> &cluster2_index_range =
      cluster2.get_index_range();

    Number cluster_distance =
      all_support_points1
        .at(internal_to_external_dof_numbering1[cluster1_index_range[0]])
        .distance(all_support_points2.at(
          internal_to_external_dof_numbering2[cluster2_index_range[0]]));
    Number point_pair_distance;

    for (types::global_dof_index index1 = cluster1_index_range[0];
         index1 < cluster1_index_range[1];
         index1++)
      {
        for (types::global_dof_index index2 = cluster2_index_range[0];
             index2 < cluster2_index_range[1];
             index2++)
          {
            point_pair_distance =
              all_support_points1
                .at(internal_to_external_dof_numbering1[index1])
                .distance(all_support_points2.at(
                  internal_to_external_dof_numbering2[index2]));

            if (point_pair_distance < cluster_distance)
              {
                cluster_distance = point_pair_distance;
              }
          }
      }

    return cluster_distance;
  }


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * the mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * @param cluster1
   * @param cluster2
   * @param all_support_points A list of support point coordinates which are ordered by DoF indices.
   * @param cell_size_at_dofs The list of estimated cell size values at DoF support points.
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number>            &cluster1,
    const Cluster<spacedim, Number>            &cluster2,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs)
  {
    /**
     * Calculate the uncorrected cluster distance.
     */
    Number uncorrected_cluster_distance =
      calc_cluster_distance(cluster1, cluster2, all_support_points);

    /**
     * Get the maximum diameter of the support sets for different DoFs, which is
     * 2 times of the cell size associated with the corresponding DoF.
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


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * the mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters share a same internal DoF numbering, which need to be mapped to
   * the external numbering for accessing the list of support point coordinates
   * as well as estimated cell sizes.}
   *
   * @param cluster1
   * @param cluster2
   * @param internal_to_external_dof_numbering
   * @param all_support_points
   * @param cell_size_at_dofs
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number> &cluster1,
    const Cluster<spacedim, Number> &cluster2,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs)
  {
    /**
     * Calculate the uncorrected cluster distance.
     */
    Number uncorrected_cluster_distance =
      calc_cluster_distance(cluster1,
                            cluster2,
                            internal_to_external_dof_numbering,
                            all_support_points);

    /**
     * Get the maximum diameter of the support sets for different DoFs, which is
     * 2 times of the cell size associated with the corresponding DoF.
     */
    Number                                        max_dof_cell_size = 0.0;
    const std::array<types::global_dof_index, 2> &cluster1_index_range =
      cluster1.get_index_range();
    const std::array<types::global_dof_index, 2> &cluster2_index_range =
      cluster2.get_index_range();

    for (types::global_dof_index index1 = cluster1_index_range[0];
         index1 < cluster1_index_range[1];
         index1++)
      {
        if (cell_size_at_dofs.at(internal_to_external_dof_numbering[index1]) >
            max_dof_cell_size)
          {
            max_dof_cell_size =
              cell_size_at_dofs.at(internal_to_external_dof_numbering[index1]);
          }
      }

    for (types::global_dof_index index2 = cluster2_index_range[0];
         index2 < cluster2_index_range[1];
         index2++)
      {
        if (cell_size_at_dofs.at(internal_to_external_dof_numbering[index2]) >
            max_dof_cell_size)
          {
            max_dof_cell_size =
              cell_size_at_dofs.at(internal_to_external_dof_numbering[index2]);
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


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * the mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * \mynote{The index sets held by the two clusters refer to two different
   * external DoF numberings.}
   *
   * @param cluster1
   * @param cluster2
   * @param all_support_points1
   * @param all_support_points2
   * @param cell_size_at_dofs1
   * @param cell_size_at_dofs2
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number>            &cluster1,
    const Cluster<spacedim, Number>            &cluster2,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2,
    const std::vector<Number>                  &cell_size_at_dofs1,
    const std::vector<Number>                  &cell_size_at_dofs2)
  {
    /**
     * Calculate the uncorrected cluster distance.
     */
    Number uncorrected_cluster_distance = calc_cluster_distance(
      cluster1, cluster2, all_support_points1, all_support_points2);

    /**
     * Get the maximum diameter of the support sets for different DoFs, which is
     * 2 times of the cell size associated with the corresponding DoF.
     */
    Number max_dof_cell_size = 0.0;

    for (const auto &index : cluster1.get_index_set())
      {
        if (cell_size_at_dofs1.at(index) > max_dof_cell_size)
          {
            max_dof_cell_size = cell_size_at_dofs1.at(index);
          }
      }

    for (const auto &index : cluster2.get_index_set())
      {
        if (cell_size_at_dofs2.at(index) > max_dof_cell_size)
          {
            max_dof_cell_size = cell_size_at_dofs2.at(index);
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


  /**
   * Calculate the minimum distance between two clusters. This calculation has
   * the mesh size correction.
   *
   * The calculation is based on measuring the distance between each pair of
   * support points contained in the clusters, which prevents the distance
   * calculation between two support sets.
   *
   * \mynote{The index sets inferred from the index ranges held by the two
   * clusters refer to two different internal DoF numberings, which need to be
   * mapped to their corresponding external numberings for accessing the list of
   * support point coordinates as well as estimated cell sizes.}
   *
   * @param cluster1
   * @param cluster2
   * @param internal_to_external_dof_numbering1
   * @param internal_to_external_dof_numbering2
   * @param all_support_points1
   * @param all_support_points2
   * @param cell_size_at_dofs1
   * @param cell_size_at_dofs2
   * @return
   */
  template <int spacedim, typename Number = double>
  Number
  calc_cluster_distance(
    const Cluster<spacedim, Number> &cluster1,
    const Cluster<spacedim, Number> &cluster2,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering1,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering2,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2,
    const std::vector<Number>                  &cell_size_at_dofs1,
    const std::vector<Number>                  &cell_size_at_dofs2)
  {
    /**
     * Calculate the uncorrected cluster distance.
     */
    Number uncorrected_cluster_distance =
      calc_cluster_distance(cluster1,
                            cluster2,
                            internal_to_external_dof_numbering1,
                            internal_to_external_dof_numbering2,
                            all_support_points1,
                            all_support_points2);

    /**
     * Get the maximum diameter of the support sets for different DoFs, which is
     * 2 times of the cell size associated with the corresponding DoF.
     */
    Number                                        max_dof_cell_size = 0.0;
    const std::array<types::global_dof_index, 2> &cluster1_index_range =
      cluster1.get_index_range();
    const std::array<types::global_dof_index, 2> &cluster2_index_range =
      cluster2.get_index_range();

    for (types::global_dof_index index1 = cluster1_index_range[0];
         index1 < cluster1_index_range[1];
         index1++)
      {
        if (cell_size_at_dofs1.at(internal_to_external_dof_numbering1[index1]) >
            max_dof_cell_size)
          {
            max_dof_cell_size = cell_size_at_dofs1.at(
              internal_to_external_dof_numbering1[index1]);
          }
      }

    for (types::global_dof_index index2 = cluster2_index_range[0];
         index2 < cluster2_index_range[1];
         index2++)
      {
        if (cell_size_at_dofs2.at(internal_to_external_dof_numbering2[index2]) >
            max_dof_cell_size)
          {
            max_dof_cell_size = cell_size_at_dofs2.at(
              internal_to_external_dof_numbering2[index2]);
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
    , index_range({{0, 0}})
    , bbox()
    , diameter(0)
  {}


  template <int spacedim, typename Number>
  Cluster<spacedim, Number>::Cluster(
    const std::vector<types::global_dof_index> &index_set)
    : index_set(index_set)
    , index_range({{0, 0}})
    , bbox()
    , diameter(0)
  {}


  template <int spacedim, typename Number>
  Cluster<spacedim, Number>::Cluster(
    const std::vector<types::global_dof_index> &index_set,
    const std::vector<Point<spacedim, Number>> &all_support_points)
    : index_set(index_set)
    , index_range({{0, 0}})
    , bbox(index_set, all_support_points)
    , diameter(calc_diameter(all_support_points))
  {}

  template <int spacedim, typename Number>
  Cluster<spacedim, Number>::Cluster(
    const std::vector<types::global_dof_index> &index_set,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs)
    : index_set(index_set)
    , index_range({{0, 0}})
    , bbox(index_set, all_support_points)
    , diameter(calc_diameter(all_support_points, cell_size_at_dofs))
  {}

  template <int spacedim, typename Number>
  Cluster<spacedim, Number>::Cluster(
    const std::vector<types::global_dof_index> &index_set,
    const SimpleBoundingBox<spacedim, Number>  &bbox,
    const std::vector<Point<spacedim, Number>> &all_support_points)
    : index_set(index_set)
    , index_range({{0, 0}})
    , bbox(bbox)
    , diameter(calc_diameter(all_support_points))
  {}

  template <int spacedim, typename Number>
  Cluster<spacedim, Number>::Cluster(
    const std::vector<types::global_dof_index> &index_set,
    const SimpleBoundingBox<spacedim, Number>  &bbox,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs)
    : index_set(index_set)
    , index_range({{0, 0}})
    , bbox(bbox)
    , diameter(calc_diameter(all_support_points, cell_size_at_dofs))
  {}

  template <int spacedim, typename Number>
  Cluster<spacedim, Number>::Cluster(const Cluster<spacedim, Number> &cluster)
    : index_set(cluster.index_set)
    , index_range(cluster.index_range)
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
  std::array<types::global_dof_index, 2> &
  Cluster<spacedim, Number>::get_index_range()
  {
    return index_range;
  }


  template <int spacedim, typename Number>
  const std::array<types::global_dof_index, 2> &
  Cluster<spacedim, Number>::get_index_range() const
  {
    return index_range;
  }


  template <int spacedim, typename Number>
  void
  Cluster<spacedim, Number>::set_index_range(
    const types::global_dof_index lower_bound,
    const types::global_dof_index pass_upper_bound)
  {
    index_range[0] = lower_bound;
    index_range[1] = pass_upper_bound;

    index_set.clear();
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

    Assert(n > 0, ExcLowerRange(n, 1));

    if (n > 1)
      {
        /**
         * Calculate the number of point pairs in the cluster. Let \f$[0, 1, 2,
         * 3, 4, 5]\f$ be the indices of support points in the cluster, whose
         * pairwise inter-distance will be calculated. The calculation is only
         * needed for the marked pairs of points as shown below.
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
         */
        Number cluster_diameter = 0;
        Number point_pair_distance;

        for (unsigned int i = 0; i < (n - 1); i++)
          {
            for (unsigned int j = i + 1; j < n; j++)
              {
                point_pair_distance =
                  all_support_points.at(index_set.at(i))
                    .distance(all_support_points.at(index_set.at(j)));

                if (point_pair_distance > cluster_diameter)
                  {
                    cluster_diameter = point_pair_distance;
                  }
              }
          }

        return cluster_diameter;
      }
    else
      {
        // When there is only one point, the diameter is zero.
        return 0;
      }
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::calc_diameter(
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points) const
  {
    // Number of support points in the cluster.
    const unsigned int n = index_range[1] - index_range[0];

    Assert(n > 0, ExcLowerRange(n, 1));

    if (n > 1)
      {
        /**
         * Calculate the number of point pairs in the cluster. Let \f$[0, 1, 2,
         * 3, 4, 5]\f$ be the indices of support points in the cluster, whose
         * pairwise inter-distance will be calculated. The calculation is only
         * needed for the marked pairs of points as shown below.
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
         */
        Number cluster_diameter = 0;
        Number point_pair_distance;

        for (unsigned int i = 0; i < (n - 1); i++)
          {
            for (unsigned int j = i + 1; j < n; j++)
              {
                point_pair_distance =
                  all_support_points
                    .at(internal_to_external_dof_numbering[index_range[0] + i])
                    .distance(all_support_points.at(
                      internal_to_external_dof_numbering[index_range[0] + j]));

                if (point_pair_distance > cluster_diameter)
                  {
                    cluster_diameter = point_pair_distance;
                  }
              }
          }

        return cluster_diameter;
      }
    else
      {
        // When there is only one point, the diameter is zero.
        return 0;
      }
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::calc_diameter(
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs) const
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
  Cluster<spacedim, Number>::calc_diameter(
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs) const
  {
    Number uncorrected_diameter =
      calc_diameter(internal_to_external_dof_numbering, all_support_points);

    Number max_dof_cell_size = 0;

    for (types::global_dof_index index = index_range[0]; index < index_range[1];
         index++)
      {
        if (cell_size_at_dofs.at(internal_to_external_dof_numbering[index]) >
            max_dof_cell_size)
          {
            max_dof_cell_size =
              cell_size_at_dofs.at(internal_to_external_dof_numbering[index]);
          }
      }

    return uncorrected_diameter + 2 * max_dof_cell_size;
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster                              &cluster,
    const std::vector<Point<spacedim, Number>> &all_support_points) const
  {
    return calc_cluster_distance((*this), cluster, all_support_points);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster &cluster,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 internal_to_external_dof_numbering,
                                 all_support_points);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster                              &cluster,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 all_support_points1,
                                 all_support_points2);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster &cluster,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering1,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering2,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 internal_to_external_dof_numbering1,
                                 internal_to_external_dof_numbering2,
                                 all_support_points1,
                                 all_support_points2);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster                              &cluster,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 all_support_points,
                                 cell_size_at_dofs);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster &cluster,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 internal_to_external_dof_numbering,
                                 all_support_points,
                                 cell_size_at_dofs);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster                              &cluster,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2,
    const std::vector<Number>                  &cell_size_at_dofs1,
    const std::vector<Number>                  &cell_size_at_dofs2) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 all_support_points1,
                                 all_support_points2,
                                 cell_size_at_dofs1,
                                 cell_size_at_dofs2);
  }


  template <int spacedim, typename Number>
  Number
  Cluster<spacedim, Number>::distance_to_cluster(
    const Cluster &cluster,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering1,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering2,
    const std::vector<Point<spacedim, Number>> &all_support_points1,
    const std::vector<Point<spacedim, Number>> &all_support_points2,
    const std::vector<Number>                  &cell_size_at_dofs1,
    const std::vector<Number>                  &cell_size_at_dofs2) const
  {
    return calc_cluster_distance((*this),
                                 cluster,
                                 internal_to_external_dof_numbering1,
                                 internal_to_external_dof_numbering2,
                                 all_support_points1,
                                 all_support_points2,
                                 cell_size_at_dofs1,
                                 cell_size_at_dofs2);
  }


  template <int spacedim, typename Number>
  bool
  Cluster<spacedim, Number>::is_subset(const Cluster &cluster) const
  {
    if (this->index_set.size() > 0 && cluster.index_set.size() > 0)
      {
        return (std::includes(cluster.index_set.begin(),
                              cluster.index_set.end(),
                              this->index_set.begin(),
                              this->index_set.end()));
      }
    else
      {
        return HierBEM::is_subset(this->index_range, cluster.index_range);
      }
  }


  template <int spacedim, typename Number>
  bool
  Cluster<spacedim, Number>::is_proper_subset(const Cluster &cluster) const
  {
    if (this->index_set.size() > 0 && cluster.index_set.size() > 0)
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
    else
      {
        return is_proper_subset(this->index_range, cluster.index_range);
      }
  }


  template <int spacedim, typename Number>
  bool
  Cluster<spacedim, Number>::is_superset(const Cluster &cluster) const
  {
    if (this->index_set.size() > 0 && cluster.index_set.size() > 0)
      {
        return (std::includes(this->index_set.begin(),
                              this->index_set.end(),
                              cluster.index_set.begin(),
                              cluster.index_set.end()));
      }
    else
      {
        return is_superset(this->index_range, cluster.index_range);
      }
  }


  template <int spacedim, typename Number>
  bool
  Cluster<spacedim, Number>::is_proper_superset(const Cluster &cluster) const
  {
    if (this->index_set.size() > 0 && cluster.index_set.size() > 0)
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
    else
      {
        return HierBEM::is_proper_superset(this->index_range,
                                           cluster.index_range);
      }
  }


  template <int spacedim, typename Number>
  void
  Cluster<spacedim, Number>::intersect(
    const Cluster                        &cluster,
    std::vector<types::global_dof_index> &index_set_intersection) const
  {
    Assert(index_set.size() >= 1, ExcLowerRange(index_set.size(), 1));
    Assert(cluster.index_set.size() >= 1,
           ExcLowerRange(cluster.index_set.size(), 1));

    index_set_intersection.clear();

    std::set_intersection(this->index_set.begin(),
                          this->index_set.end(),
                          cluster.index_set.begin(),
                          cluster.index_set.end(),
                          std::back_inserter(index_set_intersection));
  }


  template <int spacedim, typename Number>
  void
  Cluster<spacedim, Number>::intersect(
    const Cluster                          &cluster,
    std::array<types::global_dof_index, 2> &index_range_intersection) const
  {
    HierBEM::intersect(this->index_range,
                       cluster.index_range,
                       index_range_intersection);
  }


  template <int spacedim, typename Number>
  bool
  Cluster<spacedim, Number>::has_intersection(const Cluster &cluster) const
  {
    if (this->index_set.size() > 0 && cluster.index_set.size() > 0)
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
    else
      {
        std::array<types::global_dof_index, 2> index_range_intersection;
        this->intersect(cluster, index_range_intersection);

        if (index_range_intersection[1] - index_range_intersection[0] > 0)
          {
            return true;
          }
        else
          {
            return false;
          }
      }
  }


  template <int spacedim, typename Number>
  std::size_t
  Cluster<spacedim, Number>::get_cardinality() const
  {
    if (index_set.size() > 0)
      {
        return index_set.size();
      }
    else
      {
        return index_range[1] - index_range[0];
      }
  }

  template <int spacedim, typename Number>
  bool
  Cluster<spacedim, Number>::is_large(unsigned int n_min) const
  {
    if (get_cardinality() > n_min)
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
    if (cluster1.index_set.size() > 0 && cluster2.index_set.size() > 0)
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
    else
      {
        return (cluster1.index_range == cluster2.index_range);
      }
  }
} // namespace HierBEM

/**
 * @}
 */

#endif /* INCLUDE_CLUSTER_H_ */
