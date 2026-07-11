// Copyright (C) 2024-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file sauter-quad-same-panel.cc
 * @brief Verify and demonstrate Sauter quadrature performed on a pair of cells
 * for the same panel case.
 *
 * @date 2020-11-18
 * @author Jihuan Tian
 */

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "bem/bem_tools.h"
#include "bem/bem_values.h"
#include "cad_mesh/outward_surface_normal_detector.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "mapping/mapping_info.h"
#include "platform_shared/laplace_kernels.h"
#include "quadrature/sauter_quadrature.h"

using namespace dealii;
using namespace HierBEM;

// Initialize mapping objects from the first order to the maximum.
template <int dim, int spacedim>
void
initialize_mappings(std::vector<MappingInfo<dim, spacedim> *> &mappings,
                    const unsigned int max_mapping_order)
{
  // Create different orders of mapping.
  mappings.reserve(max_mapping_order);
  for (unsigned int i = 1; i <= max_mapping_order; i++)
    mappings.push_back(new MappingInfo<dim, spacedim>(i));
}

// Release all mappings on the heap.
template <int dim, int spacedim>
void
destroy_mappings(std::vector<MappingInfo<dim, spacedim> *> &mappings)
{
  for (auto m : mappings)
    if (m != nullptr)
      delete m;
}

int
main()
{
  /**
   * Generate a single cell mesh.
   */
  const unsigned int           dim      = 2;
  const unsigned int           spacedim = 3;
  Triangulation<dim, spacedim> triangulation;

  GridGenerator::hyper_rectangle(triangulation,
                                 Point<dim>(0, 0),
                                 Point<dim>(1, 2));

  std::ofstream mesh_file("./single-cell.msh");
  GridOut       grid_out;
  grid_out.write_msh(triangulation, mesh_file);

  /**
   * Generate mapping objects and associated smart pointers to their internal
   * data. High order mapping is adopted just to make this demo non-trivial in
   * the mapping aspect.
   *
   * N.B. Two mapping objects should be defined for the pair of cells
   * \f$K_x\f$ and \f$K_y\f$ respectively, because the two sets of quadrature
   * points defined in the unit cells \f$\hat{K}_x\f$ and \f$\hat{K}_y\f$ are
   * different.
   */
  const unsigned int                        max_mapping_order = 3;
  std::vector<MappingInfo<dim, spacedim> *> mappings;
  initialize_mappings(mappings, max_mapping_order);

  const unsigned int          mapping_order = 2;
  MappingInfo<dim, spacedim> &mapping_info_test_space =
    *mappings[mapping_order - 1];
  MappingInfo<dim, spacedim> &mapping_info_ansatz_space =
    *mappings[mapping_order - 1];

  /**
   * A 2D table of mapping support points for all active cells on the highest
   * level in the triangulation.
   *
   * Dim1: cell index. Dim2: mapping support point index in a cell.
   */
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  material_id_to_mapping_index[0] = mapping_order - 1;
  Table<2, Point<spacedim>> tria_mapping_support_points;
  std::vector<unsigned int> tria_mapping_indices;
  BEMTools::compute_mapping_support_points_and_indices_for_tria(
    triangulation,
    mappings,
    material_id_to_mapping_index,
    tria_mapping_support_points,
    tria_mapping_indices);

  OutwardSurfaceNormalDetector normal_detector;

  /**
   * Create different Laplace kernel functions.
   */
  HierBEM::PlatformShared::LaplaceKernel::SingleLayerKernel<spacedim> slp;
  HierBEM::PlatformShared::LaplaceKernel::DoubleLayerKernel<spacedim> dlp;
  HierBEM::PlatformShared::LaplaceKernel::AdjointDoubleLayerKernel<spacedim>
                                                                        adlp;
  HierBEM::PlatformShared::LaplaceKernel::HyperSingularKernel<spacedim> hyper;
  HierBEM::PlatformShared::LaplaceKernel::HyperSingularKernelRegular<spacedim>
    hyper_regular;

  {
    std::cout << "=== fe-order=(dirichlet:2, neumann:2), mapping order=2 ==="
              << std::endl;
    /**
     * Generate finite element, which is shared by both test and ansatz spaces.
     */
    const unsigned int  fe_order = 2;
    FE_Q<dim, spacedim> fe(fe_order);

    /**
     * Generate Dof handler.
     */
    DoFHandler<dim, spacedim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    /**
     * Generate 4D Gauss-Legendre quadrature rules for non regular cell
     * neighboring types and 2D quadrature object for the regular cell
     * neighboring type. Even though only the common edge case is considered in
     * this testcase, all of these quadrature objects are needed to initialize
     * the @p BEMValues object.
     */
    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<2> quad_rule_for_regular(quad_order_for_regular);

    /**
     * Precalculate data tables for shape function values at quadrature points
     * in the reference cells.
     *
     * Here shape functions have two meanings:
     * 1. basis polynomials for spanning the finite element space on a cell;
     * 2. basis polynomials for approximating the mapping from the reference
     * cell to real cells.
     */
    const bool               is_surface_curl_needed = true;
    BEMValues<dim, spacedim> bem_values(fe,
                                        fe,
                                        mappings,
                                        tria_mapping_support_points,
                                        quad_rule_for_same_panel,
                                        quad_rule_for_common_edge,
                                        quad_rule_for_common_vertex,
                                        quad_rule_for_regular,
                                        is_surface_curl_needed);
    bem_values.shape_function_values_same_panel();

    /**
     * Create temporary scratch data and copy data.
     */
    PairCellWiseScratchDataForFullMatrix<dim, spacedim, double> scratch_data(
      fe,
      fe,
      mappings,
      bem_values.quad_rule_for_same_panel,
      bem_values.quad_rule_for_common_edge,
      bem_values.quad_rule_for_common_vertex,
      is_surface_curl_needed);
    PairCellWisePerTaskDataForFullMatrix<dim, spacedim, double> copy_data(fe,
                                                                          fe);

    DoFHandler<dim, spacedim>::active_cell_iterator cell_iter =
      dof_handler.begin_active();

    /**
     * Compute the Sauter quadrature for each pair of cell-local shape
     * functions.
     */
    LAPACKFullMatrixExt<double> slp_cell_matrix(fe.dofs_per_cell,
                                                fe.dofs_per_cell);

    for (unsigned int i = 0; i < slp_cell_matrix.m(); i++)
      {
        for (unsigned int j = 0; j < slp_cell_matrix.n(); j++)
          {
            slp_cell_matrix(i, j) =
              sauter_quadrature_on_one_pair_of_shape_functions(
                slp,
                1.0,
                i,
                j,
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                normal_detector,
                scratch_data,
                copy_data);
          }
      }

    std::cout << "Cell matrix for single layer potential kernel:\n";
    slp_cell_matrix.print_formatted_to_mat(std::cout, "slp", 15, true, 25);

    LAPACKFullMatrixExt<double> dlp_cell_matrix(fe.dofs_per_cell,
                                                fe.dofs_per_cell);

    for (unsigned int i = 0; i < dlp_cell_matrix.m(); i++)
      {
        for (unsigned int j = 0; j < dlp_cell_matrix.n(); j++)
          {
            dlp_cell_matrix(i, j) =
              sauter_quadrature_on_one_pair_of_shape_functions(
                dlp,
                1.0,
                i,
                j,
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                normal_detector,
                scratch_data,
                copy_data);
          }
      }

    std::cout << "Cell matrix for double layer potential kernel:\n";
    dlp_cell_matrix.print_formatted_to_mat(std::cout, "dlp", 15, true, 25);

    LAPACKFullMatrixExt<double> adlp_cell_matrix(fe.dofs_per_cell,
                                                 fe.dofs_per_cell);

    for (unsigned int i = 0; i < adlp_cell_matrix.m(); i++)
      {
        for (unsigned int j = 0; j < adlp_cell_matrix.n(); j++)
          {
            adlp_cell_matrix(i, j) =
              sauter_quadrature_on_one_pair_of_shape_functions(
                adlp,
                1.0,
                i,
                j,
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                normal_detector,
                scratch_data,
                copy_data);
          }
      }

    std::cout << "Cell matrix for adjoint double layer potential kernel:\n";
    adlp_cell_matrix.print_formatted_to_mat(std::cout, "adlp", 15, true, 25);

    LAPACKFullMatrixExt<double> hyper_cell_matrix(fe.dofs_per_cell,
                                                  fe.dofs_per_cell);

    for (unsigned int i = 0; i < hyper_cell_matrix.m(); i++)
      {
        for (unsigned int j = 0; j < hyper_cell_matrix.n(); j++)
          {
            hyper_cell_matrix(i, j) =
              sauter_quadrature_on_one_pair_of_shape_functions(
                hyper,
                1.0,
                i,
                j,
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                normal_detector,
                scratch_data,
                copy_data);
          }
      }

    std::cout << "Cell matrix for hyper-singular potential kernel:\n";
    hyper_cell_matrix.print_formatted_to_mat(std::cout, "hyper", 15, true, 25);

    LAPACKFullMatrixExt<double> hyper_regular_cell_matrix(fe.dofs_per_cell,
                                                          fe.dofs_per_cell);

    for (unsigned int i = 0; i < hyper_regular_cell_matrix.m(); i++)
      {
        for (unsigned int j = 0; j < hyper_regular_cell_matrix.n(); j++)
          {
            hyper_regular_cell_matrix(i, j) =
              sauter_quadrature_on_one_pair_of_shape_functions(
                hyper_regular,
                1.0,
                i,
                j,
                cell_iter,
                cell_iter,
                mapping_info_test_space,
                mapping_info_ansatz_space,
                bem_values,
                normal_detector,
                scratch_data,
                copy_data);
          }
      }

    std::cout
      << "Cell matrix for regularized hyper-singular potential kernel:\n";
    hyper_regular_cell_matrix.print_formatted_to_mat(
      std::cout, "hyper_regular", 15, true, 25);

    dof_handler.clear();
  }

  {
    std::cout << "=== fe-order=(dirichlet:2, neumann:1), mapping order=2 ==="
              << std::endl;

    /**
     * Generate finite element, which is shared by both test and ansatz spaces.
     */
    FE_DGQ<dim, spacedim> fe_neumann_space(1);
    FE_Q<dim, spacedim>   fe_dirichlet_space(2);

    /**
     * Generate Dof handler.
     */
    DoFHandler<dim, spacedim> dof_handler_neumann_space(triangulation);
    DoFHandler<dim, spacedim> dof_handler_dirichlet_space(triangulation);
    dof_handler_neumann_space.distribute_dofs(fe_neumann_space);
    dof_handler_dirichlet_space.distribute_dofs(fe_dirichlet_space);

    const unsigned int quad_order_for_same_panel    = 5;
    const unsigned int quad_order_for_common_edge   = 4;
    const unsigned int quad_order_for_common_vertex = 4;
    const unsigned int quad_order_for_regular       = 3;

    QGauss<4> quad_rule_for_same_panel(quad_order_for_same_panel);
    QGauss<4> quad_rule_for_common_edge(quad_order_for_common_edge);
    QGauss<4> quad_rule_for_common_vertex(quad_order_for_common_vertex);
    QGauss<2> quad_rule_for_regular(quad_order_for_regular);

    DoFHandler<dim, spacedim>::active_cell_iterator cell_iter_neumann_space =
      dof_handler_neumann_space.begin_active();
    DoFHandler<dim, spacedim>::active_cell_iterator cell_iter_dirichlet_space =
      dof_handler_dirichlet_space.begin_active();

    {
      const bool               is_surface_curl_needed = false;
      BEMValues<dim, spacedim> bem_values(fe_neumann_space,
                                          fe_neumann_space,
                                          mappings,
                                          tria_mapping_support_points,
                                          quad_rule_for_same_panel,
                                          quad_rule_for_common_edge,
                                          quad_rule_for_common_vertex,
                                          quad_rule_for_regular,
                                          is_surface_curl_needed);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchDataForFullMatrix<dim, spacedim, double> scratch_data(
        fe_neumann_space,
        fe_neumann_space,
        mappings,
        bem_values.quad_rule_for_same_panel,
        bem_values.quad_rule_for_common_edge,
        bem_values.quad_rule_for_common_vertex,
        is_surface_curl_needed);
      PairCellWisePerTaskDataForFullMatrix<dim, spacedim, double> copy_data(
        fe_neumann_space, fe_neumann_space);

      /**
       * Compute the Sauter quadrature for each pair of cell-local shape
       * functions.
       */
      LAPACKFullMatrixExt<double> slp_cell_matrix(
        fe_neumann_space.dofs_per_cell, fe_neumann_space.dofs_per_cell);

      for (unsigned int i = 0; i < slp_cell_matrix.m(); i++)
        {
          for (unsigned int j = 0; j < slp_cell_matrix.n(); j++)
            {
              slp_cell_matrix(i, j) =
                sauter_quadrature_on_one_pair_of_shape_functions(
                  slp,
                  1.0,
                  i,
                  j,
                  cell_iter_neumann_space,
                  cell_iter_neumann_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  normal_detector,
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for single layer potential kernel:\n";
      slp_cell_matrix.print_formatted_to_mat(std::cout, "slp", 15, true, 25);
    }

    {
      const bool               is_surface_curl_needed = false;
      BEMValues<dim, spacedim> bem_values(fe_neumann_space,
                                          fe_dirichlet_space,
                                          mappings,
                                          tria_mapping_support_points,
                                          quad_rule_for_same_panel,
                                          quad_rule_for_common_edge,
                                          quad_rule_for_common_vertex,
                                          quad_rule_for_regular,
                                          is_surface_curl_needed);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchDataForFullMatrix<dim, spacedim, double> scratch_data(
        fe_neumann_space,
        fe_dirichlet_space,
        mappings,
        bem_values.quad_rule_for_same_panel,
        bem_values.quad_rule_for_common_edge,
        bem_values.quad_rule_for_common_vertex,
        is_surface_curl_needed);
      PairCellWisePerTaskDataForFullMatrix<dim, spacedim, double> copy_data(
        fe_neumann_space, fe_dirichlet_space);

      LAPACKFullMatrixExt<double> dlp_cell_matrix(
        fe_neumann_space.dofs_per_cell, fe_dirichlet_space.dofs_per_cell);

      for (unsigned int i = 0; i < dlp_cell_matrix.m(); i++)
        {
          for (unsigned int j = 0; j < dlp_cell_matrix.n(); j++)
            {
              dlp_cell_matrix(i, j) =
                sauter_quadrature_on_one_pair_of_shape_functions(
                  dlp,
                  1.0,
                  i,
                  j,
                  cell_iter_neumann_space,
                  cell_iter_dirichlet_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  normal_detector,
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for double layer potential kernel:\n";
      dlp_cell_matrix.print_formatted_to_mat(std::cout, "dlp", 15, true, 25);
    }

    {
      const bool               is_surface_curl_needed = false;
      BEMValues<dim, spacedim> bem_values(fe_dirichlet_space,
                                          fe_neumann_space,
                                          mappings,
                                          tria_mapping_support_points,
                                          quad_rule_for_same_panel,
                                          quad_rule_for_common_edge,
                                          quad_rule_for_common_vertex,
                                          quad_rule_for_regular,
                                          is_surface_curl_needed);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchDataForFullMatrix<dim, spacedim, double> scratch_data(
        fe_dirichlet_space,
        fe_neumann_space,
        mappings,
        bem_values.quad_rule_for_same_panel,
        bem_values.quad_rule_for_common_edge,
        bem_values.quad_rule_for_common_vertex,
        is_surface_curl_needed);
      PairCellWisePerTaskDataForFullMatrix<dim, spacedim, double> copy_data(
        fe_dirichlet_space, fe_neumann_space);

      LAPACKFullMatrixExt<double> adlp_cell_matrix(
        fe_dirichlet_space.dofs_per_cell, fe_neumann_space.dofs_per_cell);

      for (unsigned int i = 0; i < adlp_cell_matrix.m(); i++)
        {
          for (unsigned int j = 0; j < adlp_cell_matrix.n(); j++)
            {
              adlp_cell_matrix(i, j) =
                sauter_quadrature_on_one_pair_of_shape_functions(
                  adlp,
                  1.0,
                  i,
                  j,
                  cell_iter_dirichlet_space,
                  cell_iter_neumann_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  normal_detector,
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for adjoint double layer potential kernel:\n";
      adlp_cell_matrix.print_formatted_to_mat(std::cout, "adlp", 15, true, 25);
    }

    {
      const bool               is_surface_curl_needed = true;
      BEMValues<dim, spacedim> bem_values(fe_dirichlet_space,
                                          fe_dirichlet_space,
                                          mappings,
                                          tria_mapping_support_points,
                                          quad_rule_for_same_panel,
                                          quad_rule_for_common_edge,
                                          quad_rule_for_common_vertex,
                                          quad_rule_for_regular,
                                          is_surface_curl_needed);
      bem_values.shape_function_values_same_panel();

      /**
       * Create temporary scratch data and copy data.
       */
      PairCellWiseScratchDataForFullMatrix<dim, spacedim, double> scratch_data(
        fe_dirichlet_space,
        fe_dirichlet_space,
        mappings,
        bem_values.quad_rule_for_same_panel,
        bem_values.quad_rule_for_common_edge,
        bem_values.quad_rule_for_common_vertex,
        is_surface_curl_needed);
      PairCellWisePerTaskDataForFullMatrix<dim, spacedim, double> copy_data(
        fe_dirichlet_space, fe_dirichlet_space);

      LAPACKFullMatrixExt<double> hyper_cell_matrix(
        fe_dirichlet_space.dofs_per_cell, fe_dirichlet_space.dofs_per_cell);

      for (unsigned int i = 0; i < hyper_cell_matrix.m(); i++)
        {
          for (unsigned int j = 0; j < hyper_cell_matrix.n(); j++)
            {
              hyper_cell_matrix(i, j) =
                sauter_quadrature_on_one_pair_of_shape_functions(
                  hyper,
                  1.0,
                  i,
                  j,
                  cell_iter_dirichlet_space,
                  cell_iter_dirichlet_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  normal_detector,
                  scratch_data,
                  copy_data);
            }
        }

      std::cout << "Cell matrix for hyper-singular potential kernel:\n";
      hyper_cell_matrix.print_formatted_to_mat(
        std::cout, "hyper", 15, true, 25);

      LAPACKFullMatrixExt<double> hyper_regular_cell_matrix(
        fe_dirichlet_space.dofs_per_cell, fe_dirichlet_space.dofs_per_cell);

      for (unsigned int i = 0; i < hyper_regular_cell_matrix.m(); i++)
        {
          for (unsigned int j = 0; j < hyper_regular_cell_matrix.n(); j++)
            {
              hyper_regular_cell_matrix(i, j) =
                sauter_quadrature_on_one_pair_of_shape_functions(
                  hyper_regular,
                  1.0,
                  i,
                  j,
                  cell_iter_dirichlet_space,
                  cell_iter_dirichlet_space,
                  mapping_info_test_space,
                  mapping_info_ansatz_space,
                  bem_values,
                  normal_detector,
                  scratch_data,
                  copy_data);
            }
        }

      std::cout
        << "Cell matrix for regularized hyper-singular potential kernel:\n";
      hyper_regular_cell_matrix.print_formatted_to_mat(
        std::cout, "hyper_regular", 15, true, 25);
    }

    dof_handler_neumann_space.clear();
    dof_handler_dirichlet_space.clear();
  }

  destroy_mappings(mappings);
}
