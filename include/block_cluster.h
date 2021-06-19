/**
 * \file block_cluster.h
 * \brief Implementation of the class BlockCluster.
 * \ingroup hierarchical_matrices
 * \date 2021-04-20
 * \author Jihuan Tian
 */

#ifndef INCLUDE_BLOCK_CLUSTER_H_
#define INCLUDE_BLOCK_CLUSTER_H_

/**
 * \ingroup hierarchical_matrices
 * @{
 */

#include "cluster.h"
#include "cluster_tree.h"

/**
 * \brief Class for block cluster.
 *
 * A block cluster is a Cartesian product of two clusters \f$\tau\f$ and
 * \f$\sigma\f$ from two cluster trees \f$T(I)\f$ and \f$T(J)\f$, i.e. \f$\tau
 * \times \sigma\f$. This class contains pointers to the cluster tree nodes
 * which hold the data of the two clusters. Because the BlockCluster class only
 * holds pointers to nodes in cluster trees and the ClusterTree class has its
 * own memory management, the BlockCluster class does not need a destroyer.
 */
template <int spacedim, typename Number = double>
class BlockCluster
{
public:
  /**
   * Print out the block cluster data.
   * @param out
   * @param block_cluster
   * @return
   */
  template <int spacedim1, typename Number1>
  friend std::ostream &
  operator<<(std::ostream &                          out,
             const BlockCluster<spacedim1, Number1> &block_cluster);

  /**
   * Default constructor.
   */
  BlockCluster();

  /**
   * Construct from two pointers associated with the nodes in the cluster trees
   * \f$T(I)\f$ and \f$T(J)\f$.
   *
   * @param tau_node The pointer associated to the node in the cluster tree \f$T(I)\f$,
   * which holds the cluster \f$\tau\f$.
   * @param sigma_node The pointer associated to the node in the cluster tree \f$T(J)\f$,
   * which holds the cluster \f$\sigma\f$.
   */
  BlockCluster(
    typename ClusterTree<spacedim, Number>::node_pointer_type tau_node,
    typename ClusterTree<spacedim, Number>::node_pointer_type sigma_node);

  /**
   * Determine if the block cluster belongs the near field set.
   *
   * When both contained clusters are large, the block cluster is considered as
   * large.
   * @param n_min The size threshold value for determining if a cluster is large.
   * @return
   */
  void
  check_is_near_field(unsigned int n_min);

  /**
   * Determine if the block cluster is admissible. The admissibility condition
   * is evaluated without mesh cell size correction.
   *
   * @param eta Admissibility constant.
   * @return
   */
  void
  check_is_admissible(
    Number                                      eta,
    const std::vector<Point<spacedim, Number>> &all_support_points);

  /**
   * Determine if the block cluster is admissible. The admissibility condition
   * is evaluated with mesh cell size correction.
   *
   * @param eta Admissibility constant.
   * @return
   */
  void
  check_is_admissible(
    Number                                      eta,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number> &                 cell_size_at_dofs);

  /**
   * Determine if the block cluster is either admissible or small. The
   * admissibility condition is evaluated without mesh cell size correction.
   *
   * @param eta Admissibility constant.
   * @param n_min The size threshold value for determining if a cluster is large.
   * @return
   */
  bool
  is_admissible_or_small(
    Number                                      eta,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    unsigned int                                n_min);

  /**
   * Determine if the block cluster is either admissible or small. The
   * admissibility condition is evaluated with mesh cell size correction.
   *
   * @param eta Admissibility constant.
   * @param n_min The size threshold value for determining if a cluster is large.
   * @return
   */
  bool
  is_admissible_or_small(
    Number                                      eta,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number> &                 cell_size_at_dofs,
    unsigned int                                n_min);

  /**
   * Get the boolean value whether the block cluster is near field.
   */
  bool
  get_is_near_field() const;

  /**
   * Get the bollean value whether the block cluster is admissible.
   */
  bool
  get_is_admissible() const;

  typename ClusterTree<spacedim, Number>::node_pointer_type
  get_tau_node();

  typename ClusterTree<spacedim, Number>::node_const_pointer_type
  get_tau_node() const;

  typename ClusterTree<spacedim, Number>::node_pointer_type
  get_sigma_node();

  typename ClusterTree<spacedim, Number>::node_const_pointer_type
  get_sigma_node() const;

private:
  /**
   * Pointer to a node in the binary tree which holds the cluster \f$\tau\f$.
   */
  typename ClusterTree<spacedim, Number>::node_pointer_type tau_node;

  /**
   * Pointer to a node in the binary tree which holds the cluster \f$\sigma\f$.
   */
  typename ClusterTree<spacedim, Number>::node_pointer_type sigma_node;

  /**
   * The distance between cluster \f$\tau\f$ and cluster \f$\sigma\f$.
   */
  Number cluster_distance;

  /**
   * Whether the block cluster is of near-field, so that it needs a full matrix
   * representation. Otherwise, it requires a rank-r matrix representation.
   */
  bool is_near_field;

  /**
   * Is admissible.
   */
  bool is_admissible;
};

template <int spacedim, typename Number>
std::ostream &
operator<<(std::ostream &                        out,
           const BlockCluster<spacedim, Number> &block_cluster)
{
  out << "** Component tau\n";
  out << "size: "
      << block_cluster.tau_node->get_data_pointer()->get_cardinality() << "\n";
  out << *block_cluster.tau_node->get_data_pointer() << std::endl;

  out << "** Component sigma\n";
  out << "size: "
      << block_cluster.sigma_node->get_data_pointer()->get_cardinality()
      << "\n";
  out << *block_cluster.sigma_node->get_data_pointer() << std::endl;

  out << "** Distance between tau and sigma: " << block_cluster.cluster_distance
      << std::endl;

  out << "** Is near field: " << (block_cluster.is_near_field ? 1 : 0)
      << std::endl;

  out << "** Admissible state: "
      << (block_cluster.is_admissible ? "admissible" : "inadmissible")
      << std::endl;

  return out;
}

template <int spacedim, typename Number>
BlockCluster<spacedim, Number>::BlockCluster()
  : tau_node(nullptr)
  , sigma_node(nullptr)
  , cluster_distance(0)
  , is_near_field(true)
  , is_admissible(false)
{}

template <int spacedim, typename Number>
BlockCluster<spacedim, Number>::BlockCluster(
  typename ClusterTree<spacedim, Number>::node_pointer_type tau_node,
  typename ClusterTree<spacedim, Number>::node_pointer_type sigma_node)
  : tau_node(tau_node)
  , sigma_node(sigma_node)
  , cluster_distance(0)
  , is_near_field(true)
  , is_admissible(false)
{}

template <int spacedim, typename Number>
void
BlockCluster<spacedim, Number>::check_is_near_field(unsigned int n_min)
{
  if (tau_node->get_data_pointer()->is_large(n_min) &&
      sigma_node->get_data_pointer()->is_large(n_min))
    {
      is_near_field = false;
    }
  else
    {
      is_near_field = true;
    }
}

template <int spacedim, typename Number>
void
BlockCluster<spacedim, Number>::check_is_admissible(
  Number                                      eta,
  const std::vector<Point<spacedim, Number>> &all_support_points)
{
  cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
    (*sigma_node->get_data_pointer()), all_support_points);

  if (std::min(tau_node->get_data_pointer()->get_diameter(),
               sigma_node->get_data_pointer()->get_diameter()) <=
      eta * cluster_distance)
    {
      is_admissible = true;
    }
  else
    {
      is_admissible = false;
    }
}

template <int spacedim, typename Number>
void
BlockCluster<spacedim, Number>::check_is_admissible(
  Number                                      eta,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs)
{
  cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
    (*sigma_node->get_data_pointer()), all_support_points, cell_size_at_dofs);

  /**
   * N.B. The contained clusters \f$\tau\f$ and \f$\sigma\f$ in the block
   * cluster should be created with the parameter \p cell_size_at_dofs. In this
   * way, the returned cluster diameter is calculated with mesh cell size
   * correction. This is achieved when creating the two cluster trees.
   */
  if (std::min(tau_node->get_data_pointer()->get_diameter(),
               sigma_node->get_data_pointer()->get_diameter()) <=
      eta * cluster_distance)
    {
      is_admissible = true;
    }
  else
    {
      is_admissible = false;
    }
}

template <int spacedim, typename Number>
bool
BlockCluster<spacedim, Number>::is_admissible_or_small(
  Number                                      eta,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  unsigned int                                n_min)
{
  check_is_admissible(eta, all_support_points);
  check_is_near_field(n_min);

  if (is_admissible || is_near_field)
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
BlockCluster<spacedim, Number>::is_admissible_or_small(
  Number                                      eta,
  const std::vector<Point<spacedim, Number>> &all_support_points,
  const std::vector<Number> &                 cell_size_at_dofs,
  unsigned int                                n_min)
{
  check_is_admissible(eta, all_support_points, cell_size_at_dofs);
  check_is_near_field(n_min);

  if (is_admissible || is_near_field)
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
BlockCluster<spacedim, Number>::get_is_near_field() const
{
  return is_near_field;
}


template <int spacedim, typename Number>
bool
BlockCluster<spacedim, Number>::get_is_admissible() const
{
  return is_admissible;
}


template <int spacedim, typename Number>
typename ClusterTree<spacedim, Number>::node_pointer_type
BlockCluster<spacedim, Number>::get_tau_node()
{
  return tau_node;
}

template <int spacedim, typename Number>
typename ClusterTree<spacedim, Number>::node_const_pointer_type
BlockCluster<spacedim, Number>::get_tau_node() const
{
  return tau_node;
}

template <int spacedim, typename Number>
typename ClusterTree<spacedim, Number>::node_pointer_type
BlockCluster<spacedim, Number>::get_sigma_node()
{
  return sigma_node;
}

template <int spacedim, typename Number>
typename ClusterTree<spacedim, Number>::node_const_pointer_type
BlockCluster<spacedim, Number>::get_sigma_node() const
{
  return sigma_node;
}

/**
 * @}
 */

#endif /* INCLUDE_BLOCK_CLUSTER_H_ */
