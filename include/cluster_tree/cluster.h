// Copyright (C) 2021-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file cluster.h
 * @brief Implementation of the class Cluster.
 * @ingroup hierarchical_matrices
 * @date 2021-04-18
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_CLUSTER_TREE_CLUSTER_H_
#define HIERBEM_INCLUDE_CLUSTER_TREE_CLUSTER_H_

/**
 * \ingroup hierarchical_matrices
 * @{
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <utility>
#include <vector>

#include "config.h"
#include "simple_bounding_box.h"
#include "utilities/generic_functors.h"

HBEM_NS_OPEN

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
   * Construct from an index set only without support points and associated
   * bounding box.
   *
   * @param index_set
   */
  Cluster(const std::vector<types::global_dof_index> &index_set);

  /**
   * Construct from an index set in the external DoF numbering with a bounding
   * box computed from corresponding support points but without support set
   * diameter correction.
   *
   * @param index_set A list of DoF indices in the external numbering.
   * @param all_support_points A list of support points associated with a
   * function space, which may be constructed on the whole triangulation or a
   * subdomain.
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const std::vector<Point<spacedim, Number>> &all_support_points);

  /**
   * Construct from an index set in the external DoF numbering with a bounding
   * box computed from corresponding support points. Support set diameter
   * correction is also applied.
   *
   * @param index_set A list of DoF indices in the external numbering.
   * @param all_support_points A list of support points associated with a
   * function space, which may be constructed on the whole triangulation or a
   * subdomain.
   * @param dof_support_set_diameters A list of support set diameters for basis
   * functions at support points.
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const std::vector<Point<spacedim, Number>> &all_support_points,
          const std::vector<Number> &dof_support_set_diameters);

  /**
   * Construct from an index set in the external DoF numbering and a bounding
   * box without support set diameter correction.
   *
   * The input bounding box will be copied into the cluster without
   * recalculation. However, the diameter of the cluster is recalculated
   * according to the bounding box.
   *
   * @param index_set A list of DoF indices in the external numbering.
   * @param bbox Bounding box of the DoF support points associated with the list
   * of DoF indices.
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const SimpleBoundingBox<spacedim, Number>  &bbox);

  /**
   * Construct from an index set and a bounding box with support set diameter
   * correction.
   *
   * The input bounding box will be copied into the cluster without
   * recalculation. The diameter of the cluster is recalculated according to
   * this bounding box.
   *
   * @param index_set A list of DoF indices in the external numbering.
   * @param bbox Bounding box of the DoF support points associated with the list
   * of DoF indices.
   * @param dof_support_set_diameters A list of support set diameters for basis
   * functions at support points.
   */
  Cluster(const std::vector<types::global_dof_index> &index_set,
          const SimpleBoundingBox<spacedim, Number>  &bbox,
          const std::vector<Number> &dof_support_set_diameters);

  /**
   * Copy constructor.
   */
  Cluster(const Cluster<spacedim, Number> &cluster);

  /**
   * Get the reference to the index set in the external DoF numbering.
   */
  std::vector<types::global_dof_index> &
  get_index_set()
  {
    return index_set;
  }

  /**
   * Get the reference to the index set (const version) in the external DoF
   * numbering.
   */
  const std::vector<types::global_dof_index> &
  get_index_set() const
  {
    return index_set;
  }

  /**
   * Get the reference to the index range, which is in the internal numbering.
   */
  std::array<types::global_dof_index, 2> &
  get_index_range()
  {
    return index_range;
  }

  /**
   * Get the reference to the index range, which is in the internal numbering
   * (const version).
   */
  const std::array<types::global_dof_index, 2> &
  get_index_range() const
  {
    return index_range;
  }

  /**
   * Set the index range, which is in the internal numbering. After the index
   * range is set, the original index set in the external numbering will be
   * immediately cleared for saving memory.
   *
   * @param lower_bound
   * @param upper_bound
   */
  void
  set_index_range(const types::global_dof_index lower_bound,
                  const types::global_dof_index pass_upper_bound)
  {
    index_range[0] = lower_bound;
    index_range[1] = pass_upper_bound;

    index_set.clear();
  }

  /**
   * Get the reference to the bounding box.
   */
  SimpleBoundingBox<spacedim, Number> &
  get_bounding_box()
  {
    return bbox;
  }

  /**
   * Get the reference to the bounding box (const version).
   */
  const SimpleBoundingBox<spacedim, Number> &
  get_bounding_box() const
  {
    return bbox;
  }

  /**
   * Get the diameter of the cluster.
   */
  Number
  get_diameter() const
  {
    return diameter;
  }

  /**
   * Get the maximum DoF support set diameter in the cluster.
   */
  Number
  get_max_dof_support_set_diameter() const
  {
    return max_dof_support_set_diameter;
  }

  /**
   * Calculate the diameter of the cluster without support set diameter
   * correction, which is equal to the diameter of the axis-parallel bounding
   * box of the support points.
   */
  void
  calc_diameter();

  /**
   * Calculate the diameter of the cluster with support set diameter correction.
   *
   * N.B. For the finite element @p FE_Q, doubled estimated cell size is
   * adopted as an approximation of the support set diameter \f${\rm
   * diam}(Q_j)\f$ for the j-th DoF. For the finite element @p FE_DGQ, there is
   * no doubling.
   *
   * The correction is calculated as
   * \f[
   * \widetilde{\rm diam}(\tau) := {\rm diam}(\hat{Q}_{\tau}) +
   * \max_{j \in \tau} {\rm diam}(Q_j),
   * \f]
   * where \f${\rm diam}(\hat{Q}_{\tau})\f$ is the diameter of the axis-parallel
   * bounding box.
   *
   * @param dof_support_set_diameters A list of diameter values of support sets
   * for all DoFs, which is accessed with DoF indices in the external numbering.
   */
  void
  calc_diameter(const std::vector<Number> &dof_support_set_diameters);

  /**
   * Calculate the diameter of the cluster with support set diameter correction.
   *
   * \mynote{In this version, the index range in the internal DoF numbering is
   * used instead of the index set in the external numbering.}
   *
   * @param internal_to_external_dof_numbering The map from internal DoF indices
   * to external indices.
   * @param dof_support_set_diameters A list of diameter values of support sets
   * for all DoFs, which is accessed with DoF indices in the external numbering.
   */
  void
  calc_diameter(const std::vector<types::global_dof_index>
                                          &internal_to_external_dof_numbering,
                const std::vector<Number> &dof_support_set_diameters);

  /**
   * Calculate the minimum distance from the current cluster to the given
   * cluster with support set diameter correction.
   */
  Number
  distance_to_cluster(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a subset of that of the
   * given cluster.
   *
   * \mynote{The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.}
   *
   * @param cluster
   */
  bool
  is_subset(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a proper subset of that
   * of the given cluster.
   *
   * \mynote{The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.}
   *
   * @param cluster
   */
  bool
  is_proper_subset(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a superset of that of
   * the given cluster.
   *
   * \mynote{The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.}
   *
   * @param cluster
   */
  bool
  is_superset(const Cluster &cluster) const;

  /**
   * Check if the index set of the current cluster is a proper superset of
   * that of the given cluster.
   *
   * \mynote{The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.}
   *
   * @param cluster
   */
  bool
  is_proper_superset(const Cluster &cluster) const;

  /**
   * Calculate the intersection of the index sets of the current and the given
   * clusters.
   *
   * \mynote{The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.}
   *
   * @param cluster
   * @param index_set_intersection
   */
  void
  intersect(const Cluster                        &cluster,
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
   * \mynote{The index sets associated with clusters should be sorted before
   * calling this function. In the current implementation of cluster tree
   * construction, all the index sets have already been sorted.}
   *
   * @param cluster
   */
  bool
  has_intersection(const Cluster &cluster) const;

  /**
   * Get the cardinality of the index set.
   */
  std::size_t
  get_cardinality() const;

  /**
   * Determine if the cluster is large enough.
   *
   * @param n_min The size threshold value for determining if a cluster is
   * large.
   * @return
   */
  bool
  is_large(unsigned int n_min) const;

  /**
   * Estimate the memory consumption of the cluster.
   */
  std::size_t
  memory_consumption() const;

private:
  /**
   * The list of DoF indices in the external numbering in the cluster.
   */
  std::vector<types::global_dof_index> index_set;
  /**
   * The DoF index range in the internal numbering, which is a half-closed
   * half-open range.
   */
  std::array<types::global_dof_index, 2> index_range;
  /**
   * Axis-parallel bounding box holding support points in the cluster.
   */
  SimpleBoundingBox<spacedim, Number> bbox;
  /**
   * Cluster diameter.
   */
  Number diameter;
  /**
   * Maximum support set diameter for DoFs in the cluster.
   */
  Number max_dof_support_set_diameter;
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
    out << index << " ";
  out << "]\n";
  out << "Index range (internal numbering): [" << cluster.index_range[0] << " "
      << cluster.index_range[1] << ")\n";
  out << "Bounding box: " << cluster.bbox;
  out << "Diameter: " << cluster.diameter
      << "\nMaximum DoF support set diameter: "
      << cluster.max_dof_support_set_diameter;

  return out;
}


template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster()
  : index_set(0)
  , index_range({{0, 0}})
  , bbox()
  , diameter(0)
  , max_dof_support_set_diameter(0)
{}


template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set)
  : index_set(index_set)
  , index_range({{0, 0}})
  , bbox()
  , diameter(0)
  , max_dof_support_set_diameter(0)
{}


template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const std::vector<Point<spacedim, Number>> &all_support_points)
  : index_set(index_set)
  , index_range({{0, 0}})
  , bbox(index_set, all_support_points)
  , diameter(0)
  , max_dof_support_set_diameter(0)
{
  calc_diameter();
}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number>                  &dof_support_set_diameters)
  : index_set(index_set)
  , index_range({{0, 0}})
  , bbox(index_set, all_support_points)
  , diameter(0)
  , max_dof_support_set_diameter(0)
{
  calc_diameter(dof_support_set_diameters);
}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const SimpleBoundingBox<spacedim, Number>  &bbox)
  : index_set(index_set)
  , index_range({{0, 0}})
  , bbox(bbox)
  , diameter(0)
  , max_dof_support_set_diameter(0)
{
  calc_diameter();
}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(
  const std::vector<types::global_dof_index> &index_set,
  const SimpleBoundingBox<spacedim, Number>  &bbox,
  const std::vector<Number>                  &dof_support_set_diameters)
  : index_set(index_set)
  , index_range({{0, 0}})
  , bbox(bbox)
  , diameter(0)
  , max_dof_support_set_diameter(0)
{
  calc_diameter(dof_support_set_diameters);
}

template <int spacedim, typename Number>
Cluster<spacedim, Number>::Cluster(const Cluster<spacedim, Number> &cluster)
  : index_set(cluster.index_set)
  , index_range(cluster.index_range)
  , bbox(cluster.bbox)
  , diameter(cluster.diameter)
  , max_dof_support_set_diameter(cluster.max_dof_support_set_diameter)
{}


template <int spacedim, typename Number>
void
Cluster<spacedim, Number>::calc_diameter()
{
  if (index_set.size() > 1)
    diameter = bbox.diameter();
  else
    diameter = 0;
}


template <int spacedim, typename Number>
void
Cluster<spacedim, Number>::calc_diameter(
  const std::vector<Number> &dof_support_set_diameters)
{
  calc_diameter();
  max_dof_support_set_diameter = 0;
  for (const auto &index : index_set)
    {
      if (dof_support_set_diameters[index] > max_dof_support_set_diameter)
        max_dof_support_set_diameter = dof_support_set_diameters[index];
    }

  diameter += max_dof_support_set_diameter;
}


template <int spacedim, typename Number>
void
Cluster<spacedim, Number>::calc_diameter(
  const std::vector<types::global_dof_index>
                            &internal_to_external_dof_numbering,
  const std::vector<Number> &dof_support_set_diameters)
{
  calc_diameter();
  max_dof_support_set_diameter = 0;
  for (types::global_dof_index index = index_range[0]; index < index_range[1];
       index++)
    {
      if (dof_support_set_diameters[internal_to_external_dof_numbering[index]] >
          max_dof_support_set_diameter)
        {
          max_dof_support_set_diameter = dof_support_set_diameters
            [internal_to_external_dof_numbering[index]];
        }
    }

  diameter += max_dof_support_set_diameter;
}


template <int spacedim, typename Number>
Number
Cluster<spacedim, Number>::distance_to_cluster(const Cluster &cluster) const
{
  return std::max(bbox.distance_to_bounding_box(cluster.bbox) -
                    std::max(max_dof_support_set_diameter,
                             cluster.max_dof_support_set_diameter),
                  Number(0.));
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
        return true;
      else
        return false;
    }
  else
    {
      std::array<types::global_dof_index, 2> index_range_intersection;
      this->intersect(cluster, index_range_intersection);

      if (index_range_intersection[1] - index_range_intersection[0] > 0)
        return true;
      else
        return false;
    }
}


template <int spacedim, typename Number>
std::size_t
Cluster<spacedim, Number>::get_cardinality() const
{
  if (index_set.size() > 0)
    return index_set.size();
  else
    return index_range[1] - index_range[0];
}

template <int spacedim, typename Number>
bool
Cluster<spacedim, Number>::is_large(unsigned int n_min) const
{
  if (get_cardinality() > n_min)
    return true;
  else
    return false;
}


template <int spacedim, typename Number>
std::size_t
Cluster<spacedim, Number>::memory_consumption() const
{
  return sizeof(*this) +
         (dealii::MemoryConsumption::memory_consumption(index_set) -
          sizeof(index_set)) +
         (dealii::MemoryConsumption::memory_consumption(index_range) -
          sizeof(index_range)) +
         (bbox.memory_consumption() - sizeof(bbox));
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

/**
 * @}
 */

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CLUSTER_TREE_CLUSTER_H_
