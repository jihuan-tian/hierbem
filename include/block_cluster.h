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

namespace HierBEM
{
  using namespace dealii;

  /**
   * \brief Class for block cluster.
   *
   * A block cluster is a Cartesian product of two clusters \f$\tau\f$ and
   * \f$\sigma\f$ from two cluster trees \f$T(I)\f$ and \f$T(J)\f$, i.e. \f$\tau
   * \times \sigma\f$. This class contains pointers to the cluster tree nodes
   * which hold the data of the two clusters. Because the BlockCluster class
   * only holds pointers to nodes in cluster trees and the ClusterTree class has
   * its own memory management, the BlockCluster class does not need a
   * destroyer.
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
    operator<<(std::ostream                           &out,
               const BlockCluster<spacedim1, Number1> &block_cluster);

    template <int spacedim1, typename Number1>
    friend bool
    operator==(const BlockCluster<spacedim1, Number1> &block_cluster1,
               const BlockCluster<spacedim1, Number1> &block_cluster2);

    template <int spacedim1, typename Number1>
    friend bool
    is_equal(const BlockCluster<spacedim, Number> &block_cluster1,
             const BlockCluster<spacedim, Number> &block_cluster2);

    /**
     * Default constructor.
     */
    BlockCluster();

    /**
     * Construct from two pointers associated with the nodes in the cluster
     * trees \f$T(I)\f$ and \f$T(J)\f$.
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
     * Check if the index set of the current block cluster is a subset of that
     * of the given block cluster.
     * @param block_cluster
     * @return
     */
    bool
    is_subset(const BlockCluster &block_cluster) const;

    /**
     * Check if the index set of the current block cluster is a proper subset of
     * that of the given block cluster.
     * @param block_cluster
     * @return
     */
    bool
    is_proper_subset(const BlockCluster &block_cluster) const;

    /**
     * Check if the index set of the current block cluster is a superset of that
     * of the given block cluster.
     * @param block_cluster
     * @return
     */
    bool
    is_superset(const BlockCluster &block_cluster) const;

    /**
     * Check if the index set of the current block cluster is a proper superset
     * of that of the given block cluster.
     * @param block_cluster
     * @return
     */
    bool
    is_proper_superset(const BlockCluster &block_cluster) const;

    /**
     * Calculate the intersection of the index sets of the current and the given
     * block clusters.
     */
    void
    intersect(
      const BlockCluster                   &block_cluster,
      std::vector<types::global_dof_index> &tau_index_set_intersection,
      std::vector<types::global_dof_index> &sigma_index_set_intersection) const;

    /**
     * Calculate the intersection of the index ranges of the current and the
     * given block clusters.
     *
     * @param block_cluster
     * @param tau_index_range_intersection
     * @param sigma_index_range_intersection
     */
    void
    intersect(
      const BlockCluster                     &block_cluster,
      std::array<types::global_dof_index, 2> &tau_index_range_intersection,
      std::array<types::global_dof_index, 2> &sigma_index_range_intersection)
      const;

    /**
     * Determine if the index set or index range of the current block cluster
     * has a nonempty intersection with that of the given block cluster.
     * @return
     */
    bool
    has_intersection(const BlockCluster &block_cluster) const;

    /**
     * Determine if the block cluster belongs to the near field set.
     *
     * When both contained clusters are large, the block cluster is considered
     * as large.
     * @param n_min The size threshold value for determining if a cluster is large.
     * @return
     */
    void
    check_is_near_field(unsigned int n_min);

    /**
     * Determine if the block cluster belongs to the near field set.
     * @param n_min
     * @return
     */
    bool
    is_small(unsigned int n_min);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters share a same external
     * DoF numbering.}
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
     * is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters share a same internal DoF numbering, which need to be mapped to
     * the external numbering for accessing the list of support point
     * coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     */
    void
    check_is_admissible(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters refer to two different
     * external DoF numberings.}
     *
     * @param eta
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     */
    void
    check_is_admissible(
      Number                                      eta,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters refer to two different internal DoF numberings, which need to be
     * mapped to their corresponding external numberings for accessing the list
     * of support point coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering_I
     * @param internal_to_external_dof_numbering_J
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     */
    void
    check_is_admissible(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_I,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_J,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters share a same external
     * DoF numbering.}
     *
     * @param eta Admissibility constant.
     * @param all_support_points
     * @param cell_size_at_dofs
     * @return
     */
    void
    check_is_admissible(
      Number                                      eta,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters share a same internal DoF numbering, which need to be mapped to
     * the external numbering for accessing the list of support point
     * coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @param cell_size_at_dofs
     */
    void
    check_is_admissible(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters refer to two different
     * external DoF numberings.}
     *
     * @param eta
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     * @param cell_size_at_dofs_in_I
     * @param cell_size_at_dofs_in_J
     */
    void
    check_is_admissible(
      Number                                      eta,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
      const std::vector<Number>                  &cell_size_at_dofs_in_I,
      const std::vector<Number>                  &cell_size_at_dofs_in_J);

    /**
     * Determine if the block cluster is admissible. The admissibility condition
     * is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters refer to two different internal DoF numberings, which need to be
     * mapped to their corresponding external numberings for accessing the list
     * of support point coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering_I
     * @param internal_to_external_dof_numbering_J
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     * @param cell_size_at_dofs_in_I
     * @param cell_size_at_dofs_in_J
     */
    void
    check_is_admissible(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_I,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_J,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
      const std::vector<Number>                  &cell_size_at_dofs_in_I,
      const std::vector<Number>                  &cell_size_at_dofs_in_J);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters share a same external
     * DoF numbering.}
     *
     * @param eta Admissibility constant.
     * @param all_support_points
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
     * admissibility condition is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters share a same internal DoF numbering, which need to be mapped to
     * the external numbering for accessing the list of support point
     * coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @param n_min
     * @return
     */
    bool
    is_admissible_or_small(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      unsigned int                                n_min);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters refer to two different
     * external DoF numberings.}
     *
     * @param eta
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     * @param n_min
     * @return
     */
    bool
    is_admissible_or_small(
      Number                                      eta,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
      unsigned int                                n_min);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated without mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters refer to two different internal DoF numberings, which need to be
     * mapped to their corresponding external numberings for accessing the list
     * of support point coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering_I
     * @param internal_to_external_dof_numbering_J
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     * @param n_min
     * @return
     */
    bool
    is_admissible_or_small(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_I,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_J,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
      unsigned int                                n_min);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters share a same external
     * DoF numbering.}
     *
     * @param eta Admissibility constant.
     * @param n_min The size threshold value for determining if a cluster is large.
     * @return
     */
    bool
    is_admissible_or_small(
      Number                                      eta,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs,
      unsigned int                                n_min);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters share a same internal DoF numbering, which need to be mapped to
     * the external numbering for accessing the list of support point
     * coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering
     * @param all_support_points
     * @param cell_size_at_dofs
     * @param n_min
     * @return
     */
    bool
    is_admissible_or_small(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering,
      const std::vector<Point<spacedim, Number>> &all_support_points,
      const std::vector<Number>                  &cell_size_at_dofs,
      unsigned int                                n_min);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets held by the two clusters refer to two different
     * external DoF numberings.}
     *
     * @param eta
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     * @param cell_size_at_dofs_in_I
     * @param cell_size_at_dofs_in_J
     * @param n_min
     * @return
     */
    bool
    is_admissible_or_small(
      Number                                      eta,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
      const std::vector<Number>                  &cell_size_at_dofs_in_I,
      const std::vector<Number>                  &cell_size_at_dofs_in_J,
      unsigned int                                n_min);

    /**
     * Determine if the block cluster is either admissible or small. The
     * admissibility condition is evaluated with mesh cell size correction.
     *
     * \mynote{The index sets inferred from the index ranges held by the two
     * clusters refer to two different internal DoF numberings, which need to be
     * mapped to their corresponding external numberings for accessing the list
     * of support point coordinates.}
     *
     * @param eta
     * @param internal_to_external_dof_numbering_I
     * @param internal_to_external_dof_numbering_J
     * @param all_support_points_in_I
     * @param all_support_points_in_J
     * @param cell_size_at_dofs_in_I
     * @param cell_size_at_dofs_in_J
     * @param n_min
     * @return
     */
    bool
    is_admissible_or_small(
      Number eta,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_I,
      const std::vector<types::global_dof_index>
        &internal_to_external_dof_numbering_J,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
      const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
      const std::vector<Number>                  &cell_size_at_dofs_in_I,
      const std::vector<Number>                  &cell_size_at_dofs_in_J,
      unsigned int                                n_min);

    /**
     * Get the boolean value whether the block cluster is near field.
     */
    bool
    get_is_near_field() const;

    /**
     * Set the boolean value whether the block cluster is near field.
     */
    void
    set_is_near_field(const bool is_near_field);

    /**
     * Get the boolean value whether the block cluster is admissible.
     */
    bool
    get_is_admissible() const;

    /**
     * Set the boolean value whether the block cluster is admissible.
     */
    void
    set_is_admissible(const bool is_admissible);

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
     * Pointer to a node in the binary tree which holds the cluster
     * \f$\sigma\f$.
     */
    typename ClusterTree<spacedim, Number>::node_pointer_type sigma_node;

    /**
     * The distance between cluster \f$\tau\f$ and cluster \f$\sigma\f$. Its
     * value is computed when evaluating the admissibility condition.
     */
    Number cluster_distance;

    /**
     * Whether the block cluster is of near-field, so that it needs a full
     * matrix representation. Otherwise, it requires a rank-r matrix
     * representation.
     */
    bool is_near_field;

    /**
     * Is admissible.
     */
    bool is_admissible;
  };

  template <int spacedim, typename Number>
  std::ostream &
  operator<<(std::ostream                         &out,
             const BlockCluster<spacedim, Number> &block_cluster)
  {
    out << "** Component tau\n";
    out << "size: "
        << block_cluster.tau_node->get_data_pointer()->get_cardinality()
        << "\n";
    out << *block_cluster.tau_node->get_data_pointer() << std::endl;

    out << "** Component sigma\n";
    out << "size: "
        << block_cluster.sigma_node->get_data_pointer()->get_cardinality()
        << "\n";
    out << *block_cluster.sigma_node->get_data_pointer() << std::endl;

    out << "** Distance between tau and sigma: "
        << block_cluster.cluster_distance << std::endl;

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
  bool
  BlockCluster<spacedim, Number>::is_subset(
    const BlockCluster &block_cluster) const
  {
    return (this->tau_node->get_data_reference().is_subset(
              block_cluster.tau_node->get_data_reference()) &&
            this->sigma_node->get_data_reference().is_subset(
              block_cluster.sigma_node->get_data_reference()));
  }


  template <int spacedim, typename Number>
  bool
  BlockCluster<spacedim, Number>::is_proper_subset(
    const BlockCluster &block_cluster) const
  {
    if (this->tau_node->get_data_reference().is_subset(
          block_cluster.tau_node->get_data_reference()) &&
        this->sigma_node->get_data_reference().is_subset(
          block_cluster.sigma_node->get_data_reference()))
      {
        if ((this->tau_node->get_data_reference().get_cardinality() ==
             block_cluster.tau_node->get_data_reference().get_cardinality()) &&
            (this->sigma_node->get_data_reference().get_cardinality() ==
             block_cluster.sigma_node->get_data_reference().get_cardinality()))
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
  BlockCluster<spacedim, Number>::is_superset(
    const BlockCluster &block_cluster) const
  {
    return (this->tau_node->get_data_reference().is_superset(
              block_cluster.tau_node->get_data_reference()) &&
            this->sigma_node->get_data_reference().is_superset(
              block_cluster.sigma_node->get_data_reference()));
  }


  template <int spacedim, typename Number>
  bool
  BlockCluster<spacedim, Number>::is_proper_superset(
    const BlockCluster &block_cluster) const
  {
    if (this->tau_node->get_data_reference().is_superset(
          block_cluster.tau_node->get_data_reference()) &&
        this->sigma_node->get_data_reference().is_superset(
          block_cluster.sigma_node->get_data_reference()))
      {
        if ((this->tau_node->get_data_reference().get_cardinality() ==
             block_cluster.tau_node->get_data_reference().get_cardinality()) &&
            (this->sigma_node->get_data_reference().get_cardinality() ==
             block_cluster.sigma_node->get_data_reference().get_cardinality()))
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
  BlockCluster<spacedim, Number>::intersect(
    const BlockCluster                   &block_cluster,
    std::vector<types::global_dof_index> &tau_index_set_intersection,
    std::vector<types::global_dof_index> &sigma_index_set_intersection) const
  {
    this->tau_node->get_data_reference().intersect(
      block_cluster.tau_node->get_data_reference(), tau_index_set_intersection);
    this->sigma_node->get_data_reference().intersect(
      block_cluster.sigma_node->get_data_reference(),
      sigma_index_set_intersection);
  }


  template <int spacedim, typename Number>
  void
  BlockCluster<spacedim, Number>::intersect(
    const BlockCluster                     &block_cluster,
    std::array<types::global_dof_index, 2> &tau_index_range_intersection,
    std::array<types::global_dof_index, 2> &sigma_index_range_intersection)
    const
  {
    this->tau_node->get_data_reference().intersect(
      block_cluster.tau_node->get_data_reference(),
      tau_index_range_intersection);
    this->sigma_node->get_data_reference().intersect(
      block_cluster.sigma_node->get_data_reference(),
      sigma_index_range_intersection);
  }


  template <int spacedim, typename Number>
  bool
  BlockCluster<spacedim, Number>::has_intersection(
    const BlockCluster &block_cluster) const
  {
    if (this->tau_node->get_data_reference().get_index_set().size() > 0 &&
        this->sigma_node->get_data_reference().get_index_set().size() > 0 &&
        block_cluster.tau_node->get_data_reference().get_index_set().size() >
          0 &&
        block_cluster.sigma_node->get_data_reference().get_index_set().size() >
          0)
      {
        /**
         * When the index sets are non-empty.
         */
        std::vector<types::global_dof_index> tau_index_set_intersection;
        std::vector<types::global_dof_index> sigma_index_set_intersection;

        this->intersect(block_cluster,
                        tau_index_set_intersection,
                        sigma_index_set_intersection);

        if (tau_index_set_intersection.size() > 0 &&
            sigma_index_set_intersection.size() > 0)
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
        /**
         * When the index sets are empty, we check the intersection of index
         * ranges.
         */
        std::array<types::global_dof_index, 2> tau_index_range_intersection;
        std::array<types::global_dof_index, 2> sigma_index_range_intersection;

        this->intersect(block_cluster,
                        tau_index_range_intersection,
                        sigma_index_range_intersection);

        if (((tau_index_range_intersection[1] -
              tau_index_range_intersection[0]) > 0) &&
            ((sigma_index_range_intersection[1] -
              sigma_index_range_intersection[0]) > 0))
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
  bool
  BlockCluster<spacedim, Number>::is_small(unsigned int n_min)
  {
    check_is_near_field(n_min);

    return is_near_field;
  }


  template <int spacedim, typename Number>
  void
  BlockCluster<spacedim, Number>::check_is_admissible(
    Number                                      eta,
    const std::vector<Point<spacedim, Number>> &all_support_points)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(), all_support_points);

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
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(),
      internal_to_external_dof_numbering,
      all_support_points);

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
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(),
      all_support_points_in_I,
      all_support_points_in_J);

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
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_I,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_J,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(),
      internal_to_external_dof_numbering_I,
      internal_to_external_dof_numbering_J,
      all_support_points_in_I,
      all_support_points_in_J);

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
    const std::vector<Number>                  &cell_size_at_dofs)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(), all_support_points, cell_size_at_dofs);

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
  void
  BlockCluster<spacedim, Number>::check_is_admissible(
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(),
      internal_to_external_dof_numbering,
      all_support_points,
      cell_size_at_dofs);

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
  void
  BlockCluster<spacedim, Number>::check_is_admissible(
    Number                                      eta,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
    const std::vector<Number>                  &cell_size_at_dofs_in_I,
    const std::vector<Number>                  &cell_size_at_dofs_in_J)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(),
      all_support_points_in_I,
      all_support_points_in_J,
      cell_size_at_dofs_in_I,
      cell_size_at_dofs_in_J);

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
  void
  BlockCluster<spacedim, Number>::check_is_admissible(
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_I,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_J,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
    const std::vector<Number>                  &cell_size_at_dofs_in_I,
    const std::vector<Number>                  &cell_size_at_dofs_in_J)
  {
    cluster_distance = tau_node->get_data_pointer()->distance_to_cluster(
      sigma_node->get_data_reference(),
      internal_to_external_dof_numbering_I,
      internal_to_external_dof_numbering_J,
      all_support_points_in_I,
      all_support_points_in_J,
      cell_size_at_dofs_in_I,
      cell_size_at_dofs_in_J);

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
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    unsigned int                                n_min)
  {
    check_is_admissible(eta,
                        internal_to_external_dof_numbering,
                        all_support_points);
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
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
    unsigned int                                n_min)
  {
    check_is_admissible(eta, all_support_points_in_I, all_support_points_in_J);
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
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_I,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_J,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
    unsigned int                                n_min)
  {
    check_is_admissible(eta,
                        internal_to_external_dof_numbering_I,
                        internal_to_external_dof_numbering_J,
                        all_support_points_in_I,
                        all_support_points_in_J);
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
    const std::vector<Number>                  &cell_size_at_dofs,
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
  BlockCluster<spacedim, Number>::is_admissible_or_small(
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering,
    const std::vector<Point<spacedim, Number>> &all_support_points,
    const std::vector<Number>                  &cell_size_at_dofs,
    unsigned int                                n_min)
  {
    check_is_admissible(eta,
                        internal_to_external_dof_numbering,
                        all_support_points,
                        cell_size_at_dofs);
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
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
    const std::vector<Number>                  &cell_size_at_dofs_in_I,
    const std::vector<Number>                  &cell_size_at_dofs_in_J,
    unsigned int                                n_min)
  {
    check_is_admissible(eta,
                        all_support_points_in_I,
                        all_support_points_in_J,
                        cell_size_at_dofs_in_I,
                        cell_size_at_dofs_in_J);
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
    Number eta,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_I,
    const std::vector<types::global_dof_index>
      &internal_to_external_dof_numbering_J,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_I,
    const std::vector<Point<spacedim, Number>> &all_support_points_in_J,
    const std::vector<Number>                  &cell_size_at_dofs_in_I,
    const std::vector<Number>                  &cell_size_at_dofs_in_J,
    unsigned int                                n_min)
  {
    check_is_admissible(eta,
                        internal_to_external_dof_numbering_I,
                        internal_to_external_dof_numbering_J,
                        all_support_points_in_I,
                        all_support_points_in_J,
                        cell_size_at_dofs_in_I,
                        cell_size_at_dofs_in_J);
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
  void
  BlockCluster<spacedim, Number>::set_is_near_field(const bool is_near_field)
  {
    this->is_near_field = is_near_field;
  }


  template <int spacedim, typename Number>
  bool
  BlockCluster<spacedim, Number>::get_is_admissible() const
  {
    return is_admissible;
  }


  template <int spacedim, typename Number>
  void
  BlockCluster<spacedim, Number>::set_is_admissible(const bool is_admissible)
  {
    this->is_admissible = is_admissible;
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
   * Check the equality of two block clusters by shallow comparison.
   *
   * The comparison is based on the pointer addresses to the tau cluster node
   * and the sigma cluster node. Therefore, this is a "shallow" comparison for
   * performance issue.
   *
   * <dl class="section note">
   *   <dt>Note</dt>
   *   <dd>This method is valid in the following two cases.
   *
   * 1. The two block clusters to be compared belong to a same block cluster
   * tree. In this scenario, because in either cluster tree, \f$T(I)\f$ or
   * \f$T(J)\f$, comprising the block cluster tree, the cluster nodes contained
   * are created on the heap, hence each of them has an address in the memory
   * different from all the others. Therefore, the equality of two block
   * clusters, i.e. the equality of the index sets in the \f$\tau\f$ cluster
   * nodes and the equality of the index sets in the \f$\sigma\f$ cluster nodes,
   * is equivalent to the equality of the pointer addresses of \f$\tau\f$
   * cluster nodes and the equality of the pointer addresses of \f$\sigma\f$
   * cluster nodes.
   *
   * 2. The two block clusters to be compared belong to two different block
   * cluster tress, each of which is built from the two cluster trees \f$T(I)\f$
   * and \f$T(J)\f$.
   * </dd>
   * </dl>
   *
   * @param block_cluster1
   * @param block_cluster2
   * @return
   */
  template <int spacedim, typename Number>
  bool
  operator==(const BlockCluster<spacedim, Number> &block_cluster1,
             const BlockCluster<spacedim, Number> &block_cluster2)
  {
    if ((block_cluster1.tau_node == block_cluster2.tau_node) &&
        (block_cluster1.sigma_node == block_cluster2.sigma_node))
      {
        return true;
      }
    else
      {
        return false;
      }
  }


  /**
   * Check the equality of two block cluster by comparing the contents of block
   * cluster's index sets. Compared to \p BlockCluster<spacedim,
   * Number>::operator==, this can be considered as deep comparison.
   *
   * @param block_cluster1
   * @param block_cluster2
   * @return
   */
  template <int spacedim, typename Number>
  bool
  is_equal(const BlockCluster<spacedim, Number> &block_cluster1,
           const BlockCluster<spacedim, Number> &block_cluster2)
  {
    if ((block_cluster1.get_tau_node()->get_data_reference() ==
         block_cluster2.get_tau_node()->get_data_reference()) &&
        (block_cluster1.get_sigma_node()->get_data_reference() ==
         block_cluster2.get_sigma_node()->get_data_reference()))
      {
        return true;
      }
    else
      {
        return false;
      }
  }
} // namespace HierBEM

/**
 * @}
 */

#endif /* INCLUDE_BLOCK_CLUSTER_H_ */
