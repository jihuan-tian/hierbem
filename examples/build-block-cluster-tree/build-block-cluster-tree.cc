// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file build-block-cluster-tree.cc
 * @brief Example for building a block cluster tree.
 *
 * @ingroup examples
 * @date 2025-10-13
 * @author Jihuan Tian
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "cluster_tree/block_cluster_tree.h"
#include "cluster_tree/cluster_tree.h"
#include "dofs/dof_tools_ext.h"

using namespace HierBEM;
using namespace dealii;

// Builder class for cluster tree.
template <int spacedim>
class ClusterTreeBuilder
{
public:
  template <int dim>
  ClusterTreeBuilder(const Mapping<dim, spacedim>    &mapping,
                     const DoFHandler<dim, spacedim> &dof_handler,
                     const unsigned int               _n_min);

  // Build a cluster tree and return it as a unique smart pointer, since it will
  // be associated with a unique function space.
  std::unique_ptr<ClusterTree<spacedim>>
  build() const;

  std::vector<Point<spacedim>> &
  get_support_points()
  {
    return support_points;
  }

  const std::vector<Point<spacedim>> &
  get_support_points() const
  {
    return support_points;
  }

  std::vector<double> &
  get_dof_average_cell_size()
  {
    return dof_average_cell_size;
  }

  const std::vector<double> &
  get_dof_average_cell_size() const
  {
    return dof_average_cell_size;
  }

private:
  // List of support points in the finite element associated with this cluster
  // tree.
  std::vector<Point<spacedim>> support_points;
  // List of DoF indices in the finite element, starting from zero.
  std::vector<types::global_dof_index> dof_indices;
  // List of average cell size estimated at support points.
  std::vector<double> dof_average_cell_size;
  // Minimum number of DoFs in a cluster. When the actual number of DoFs in a
  // cluster is smaller than @p n_min , the cluster is a leaf node in the
  // cluster tree.
  unsigned int n_min;
};


template <int spacedim>
template <int dim>
ClusterTreeBuilder<spacedim>::ClusterTreeBuilder(
  const Mapping<dim, spacedim>    &mapping,
  const DoFHandler<dim, spacedim> &dof_handler,
  const unsigned int               _n_min)
  : n_min(_n_min)
{
  Assert(dof_handler.get_fe().has_support_points(), ExcInternalError());

  const types::global_dof_index n_dofs = dof_handler.n_dofs();

  // Get the coordinates for all support points.
  support_points.resize(n_dofs);
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

  // Generate a list of DoF indices starting from zero.
  dof_indices.resize(n_dofs);
  for (types::global_dof_index d = 0; d < n_dofs; d++)
    dof_indices[d] = d;

  // Calculate the average mesh cell size at each support point.
  dof_average_cell_size.assign(n_dofs, 0);
  DoFToolsExt::map_dofs_to_average_cell_size(dof_handler,
                                             dof_average_cell_size);
}


template <int spacedim>
std::unique_ptr<ClusterTree<spacedim>>
ClusterTreeBuilder<spacedim>::build() const
{
  // Create a cluster tree for all the DoF indices.
  auto cluster_tree = std::make_unique<ClusterTree<spacedim>>(
    dof_indices, support_points, dof_average_cell_size, n_min);
  // Partition the cluster tree.
  cluster_tree->partition(support_points, dof_average_cell_size);

  return cluster_tree;
}


// Class for a function space used in BEM, which contains a cluster tree.
template <int dim, int spacedim>
class BEMFunctionSpace
{
public:
  BEMFunctionSpace(const DoFHandler<dim, spacedim> &dof_handler_,
                   const Mapping<dim, spacedim>    &mapping,
                   const unsigned int               n_min);

  ClusterTree<spacedim> &
  get_cluster_tree()
  {
    return *cluster_tree;
  }

  const ClusterTree<spacedim> &
  get_cluster_tree() const
  {
    return *cluster_tree;
  }

  ClusterTreeBuilder<spacedim> &
  get_cluster_tree_builder()
  {
    return *cluster_tree_builder;
  }

  const ClusterTreeBuilder<spacedim> &
  get_cluster_tree_builder() const
  {
    return *cluster_tree_builder;
  }

  std::vector<Point<spacedim>> &
  get_support_points()
  {
    return cluster_tree_builder->get_support_points();
  }

  const std::vector<Point<spacedim>> &
  get_support_points() const
  {
    return cluster_tree_builder->get_support_points();
  }

  std::vector<double> &
  get_dof_average_cell_size()
  {
    return cluster_tree_builder->get_dof_average_cell_size();
  }

  const std::vector<double> &
  get_dof_average_cell_size() const
  {
    return cluster_tree_builder->get_dof_average_cell_size();
  }

  std::vector<types::global_dof_index> &
  get_internal_to_external_dof_numbering()
  {
    return cluster_tree->get_internal_to_external_dof_numbering();
  }

  const std::vector<types::global_dof_index> &
  get_internal_to_external_dof_numbering() const
  {
    return cluster_tree->get_internal_to_external_dof_numbering();
  }

  std::vector<types::global_dof_index> &
  get_external_to_internal_dof_numbering()
  {
    return cluster_tree->get_external_to_internal_dof_numbering();
  }

  const std::vector<types::global_dof_index> &
  get_external_to_internal_dof_numbering() const
  {
    return cluster_tree->get_external_to_internal_dof_numbering();
  }

private:
  const DoFHandler<dim, spacedim>              &dof_handler;
  std::unique_ptr<ClusterTree<spacedim>>        cluster_tree;
  std::unique_ptr<ClusterTreeBuilder<spacedim>> cluster_tree_builder;
};


template <int dim, int spacedim>
BEMFunctionSpace<dim, spacedim>::BEMFunctionSpace(
  const DoFHandler<dim, spacedim> &dof_handler_,
  const Mapping<dim, spacedim>    &mapping,
  const unsigned int               n_min)
  : dof_handler(dof_handler_)
{
  cluster_tree_builder =
    std::make_unique<ClusterTreeBuilder<spacedim>>(mapping, dof_handler, n_min);
  cluster_tree = cluster_tree_builder->build();
}


// Class for a bilinear form, which involves a trial space and a test space.
template <int dim, int spacedim>
class BEMBilinearForm
{
public:
  // As a convention, the trial space is placed before the test space when we
  // define a bilinear form.
  BEMBilinearForm(const BEMFunctionSpace<dim, spacedim> &trial_space_,
                  const BEMFunctionSpace<dim, spacedim> &test_space_);

  // @param eta Admissibility constant. Englaring this parameter will make more
  // leaf nodes of the block cluster tree be far field.
  // @param n_min Minimum number of DoFs in a cluster. For a block cluster
  // \f$\tau\times \sigma\f$, whenever the cardinality of \f$\tau\f$ or
  // \f$\sigma\f$ is smaller than @p n_min , this block cluster is a near field
  // node.
  void
  build_block_cluster_tree(const double eta, const unsigned int n_min);

  ClusterTree<spacedim> &
  get_cluster_tree_trial_space()
  {
    return trial_space.get_cluster_tree();
  }

  const ClusterTree<spacedim> &
  get_cluster_tree_trial_space() const
  {
    return trial_space.get_cluster_tree();
  }

  ClusterTree<spacedim> &
  get_cluster_tree_test_space()
  {
    return test_space.get_cluster_tree();
  }

  const ClusterTree<spacedim> &
  get_cluster_tree_test_space() const
  {
    return test_space.get_cluster_tree();
  }

  BlockClusterTree<spacedim> &
  get_block_cluster_tree()
  {
    return *block_cluster_tree;
  }

  const BlockClusterTree<spacedim> &
  get_block_cluster_tree() const
  {
    return *block_cluster_tree;
  }

private:
  const BEMFunctionSpace<dim, spacedim>      &trial_space;
  const BEMFunctionSpace<dim, spacedim>      &test_space;
  std::unique_ptr<BlockClusterTree<spacedim>> block_cluster_tree;
};


template <int dim, int spacedim>
BEMBilinearForm<dim, spacedim>::BEMBilinearForm(
  const BEMFunctionSpace<dim, spacedim> &trial_space_,
  const BEMFunctionSpace<dim, spacedim> &test_space_)
  : trial_space(trial_space_)
  , test_space(test_space_)
{}


template <int dim, int spacedim>
void
BEMBilinearForm<dim, spacedim>::build_block_cluster_tree(
  const double       eta,
  const unsigned int n_min)
{
  // When building a block cluster tree, the test space appears before the trial
  // space, since the test space is related to matrix rows, while the trial
  // space is related to matrix columns.
  block_cluster_tree = std::make_unique<BlockClusterTree<spacedim>>(
    test_space.get_cluster_tree(), trial_space.get_cluster_tree(), eta, n_min);
  block_cluster_tree->partition(
    test_space.get_internal_to_external_dof_numbering(),
    trial_space.get_internal_to_external_dof_numbering(),
    test_space.get_support_points(),
    trial_space.get_support_points(),
    test_space.get_dof_average_cell_size(),
    trial_space.get_dof_average_cell_size());
}


int
main()
{
  // Generate a triangulation for a unit sphere.
  const unsigned int    dim      = 2;
  const unsigned int    spacedim = 3;
  const Point<spacedim> center(0, 0, 0);
  const double          radius(1);
  const unsigned int    refinement = 1;

  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(refinement);

  // Create a mapping object for transforming unit support points in the unit
  // cell to all real cells.
  const unsigned int            mapping_order = 2;
  const MappingQ<dim, spacedim> mapping(mapping_order);

  // Parameters for cluster trees and block cluster tree.
  const unsigned int n_min_H_half             = 4;
  const unsigned int n_min_H_minus_half       = 4;
  const double       eta                      = 4;
  const unsigned int n_min_block_cluster_tree = 4;

  // Create a continuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{1/2}(\Gamma)\f$.
  const unsigned int        fe_H_half_order = 1;
  FE_Q<dim, spacedim>       fe_H_half(fe_H_half_order);
  DoFHandler<dim, spacedim> dof_handler_H_half(tria);
  dof_handler_H_half.distribute_dofs(fe_H_half);
  BEMFunctionSpace<dim, spacedim> H_half(dof_handler_H_half,
                                         mapping,
                                         n_min_H_half);

  // Create a discontinuous Lagrangian finite element and a DoF handler for the
  // Sobolev space \f$H^{-1/2}(\Gamma)\f$ space.
  const unsigned int        fe_H_minus_half_order = 0;
  FE_DGQ<dim, spacedim>     fe_H_minus_half(fe_H_minus_half_order);
  DoFHandler<dim, spacedim> dof_handler_H_minus_half(tria);
  dof_handler_H_minus_half.distribute_dofs(fe_H_minus_half);
  BEMFunctionSpace<dim, spacedim> H_minus_half(dof_handler_H_minus_half,
                                               mapping,
                                               n_min_H_minus_half);

  // Create a bilinear form \f$b_V: H^{-1/2}(\Gamma)\times H^{-1/2}(\Gamma)
  // \rightarrow \mathbb{R}\f$ for the single layer potential operator \f$V\f$.
  BEMBilinearForm<dim, spacedim> bV(H_minus_half, H_minus_half);
  bV.build_block_cluster_tree(eta, n_min_block_cluster_tree);
  // Create a bilinear form \f$b_K: H^{1/2}(\Gamma)\times H^{-1/2}(\Gamma)
  // \rightarrow \mathbb{R}\f$ for the double layer potential operator \f$K\f$.
  BEMBilinearForm<dim, spacedim> bK(H_half, H_minus_half);
  bK.build_block_cluster_tree(eta, n_min_block_cluster_tree);

  // Print out the cluster trees for the two function spaces.
  std::ofstream graph("cluster-tree-H-half.puml");
  H_half.get_cluster_tree().print_tree_info_as_dot(graph);
  graph.close();

  graph.open("cluster-tree-H-minus-half.puml");
  H_minus_half.get_cluster_tree().print_tree_info_as_dot(graph);
  graph.close();

  // Print out the block cluster trees for the two bilinear forms.
  graph.open("bV-block-cluster-tree.puml");
  bV.get_block_cluster_tree().print_bct_info_as_dot(graph);
  graph.close();

  graph.open("bK-block-cluster-tree.puml");
  bK.get_block_cluster_tree().print_bct_info_as_dot(graph);
  graph.close();
}
