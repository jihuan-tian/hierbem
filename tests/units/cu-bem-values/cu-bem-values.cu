// Copyright (C) 2023-2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file cu-bem-values.cu
 * @brief Verify the initialization of BEMValues and precomputation of cell
 * values on the host and device.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-01
 */

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <catch2/catch_all.hpp>

#include <map>
#include <vector>

#include "bem/bem_tools.h"
#include "bem/bem_values.h"
#include "bem/cu_bem_values.hcu"
#include "cad_mesh/outward_surface_normal_detector.h"
#include "config_file/config_structs.h"
#include "config_file/cu_related.h"
#include "linear_algebra/cu_table.hcu"
#include "mapping/mapping_info.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/cu_debug_tools.h"
#include "utilities/generic_functors.h"
#include "utilities/unary_template_arg_containers.h"

using namespace Catch::Matchers;
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

TEST_CASE("Initialize BEMValues and precompute cell values", "[cuda]")
{
  ConfParallelization parallel_params;
  initCudaRuntime(parallel_params);

  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  const Point<spacedim>        center(0, 0, 0);
  const double                 radius(1);
  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_sphere(tria, center, radius);
  tria.refine_global(6);
  const types::global_cell_index n_cells = tria.n_active_cells();

  FE_Q<dim, spacedim>   fe_for_dirichlet_space(1);
  FE_DGQ<dim, spacedim> fe_for_neumann_space(0);

  // A @p DoFHandler is needed to acquire active cell iterators related to
  // @p DoFCellAccessor. Acquiring active cell iterators via a @p Triangulation
  // object is not allowed, since it involve @p CellAccessor.
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_for_dirichlet_space);

  const unsigned int                        max_mapping_order = 3;
  std::vector<MappingInfo<dim, spacedim> *> mappings;
  initialize_mappings(mappings, max_mapping_order);

  // Use second order mapping for the sphere.
  std::map<types::material_id, unsigned int> material_id_to_mapping_index;
  material_id_to_mapping_index[0] = 1;

  // Compute mapping support points and indices for all active cells in the
  // triangulation.
  Table<2, Point<spacedim>> tria_mapping_support_points_cpu;
  HierBEM::CUDAWrappers::CUDATable<2, Point<spacedim>>
                            tria_mapping_support_points_gpu;
  std::vector<unsigned int> tria_mapping_indices_cpu;
  HierBEM::CUDAWrappers::CUDATable<1, unsigned int> tria_mapping_indices_gpu;

  BEMTools::compute_mapping_support_points_and_indices_for_tria(
    tria,
    mappings,
    material_id_to_mapping_index,
    tria_mapping_support_points_cpu,
    tria_mapping_indices_cpu);

  // Copy mapping support points and indices to the device.
  tria_mapping_support_points_gpu.allocate(
    tria_mapping_support_points_cpu.size());
  tria_mapping_support_points_gpu.assign_from_host(
    tria_mapping_support_points_cpu);
  tria_mapping_indices_gpu.allocate(
    TableIndices<1>(tria_mapping_indices_cpu.size()));
  tria_mapping_indices_gpu.assign_from_host(tria_mapping_indices_cpu);

  // Create BEM values objects on the host and the device. Meanwhile, compute
  // cell values for the regular cell neighboring type. In a BEM solver using
  // full matrices, these values are only computed on the host. In a BEM solver
  // using H-matrices, these values are only computed on the device.
  //
  // Here we compute them both on the host and the device then check their
  // equality.
  SauterQuadratureRule<dim>        sauter_quad_rule(5, 4, 4, 3);
  OutwardSurfaceNormalDetector     normal_detector;
  BEMValues<dim, spacedim, double> bem_values_cpu(
    fe_for_dirichlet_space,
    fe_for_neumann_space,
    mappings,
    tria_mapping_support_points_cpu,
    sauter_quad_rule.quad_rule_for_same_panel,
    sauter_quad_rule.quad_rule_for_common_edge,
    sauter_quad_rule.quad_rule_for_common_vertex,
    sauter_quad_rule.quad_rule_for_regular,
    true);
  bem_values_cpu.fill_shape_function_value_tables();
  bem_values_cpu.compute_bilinear_form_cell_values_for_regular(
    tria, tria_mapping_indices_cpu, normal_detector);

  HierBEM::CUDAWrappers::CUDABEMValues<dim, spacedim, double> bem_values_gpu;
  bem_values_gpu.allocate_and_assign_from_host(bem_values_cpu,
                                               tria_mapping_support_points_gpu,
                                               tria_mapping_indices_gpu);

  // Generate a map from used cell indices (local) to all cell indices (local).
  std::vector<types::global_cell_index> local_to_global_cell_index_map(n_cells);
  gen_linear_indices<vector_uta, types::global_cell_index>(
    local_to_global_cell_index_map);

  // Collect pointers to all active cell iterators in the triangulation held by
  // the @p DoFHandler.
  std::vector<typename DoFHandler<dim, spacedim>::cell_iterator> cell_iterators(
    n_cells);
  std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>
                           cell_iterator_ptrs(n_cells);
  types::global_cell_index c = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_iterators[c]     = cell;
      cell_iterator_ptrs[c] = &cell_iterators[c];
      c++;
    }

  bem_values_gpu.compute_bilinear_form_cell_values_for_regular(
    cell_iterator_ptrs, local_to_global_cell_index_map, normal_detector);

  REQUIRE(is_equal(bem_values_cpu, bem_values_gpu, 1e-11));

  bem_values_gpu.release();
}
