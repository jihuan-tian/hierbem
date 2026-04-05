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
 * @brief Verify the initialization of BEMValues on GPU.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-02-01
 */

#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgq.h>

#include <catch2/catch_all.hpp>

#include <iostream>
#include <vector>

#include "bem/bem_values.hcu"
#include "bem/cu_bem_values.hcu"
#include "mapping/mapping_info.h"
#include "quadrature/sauter_quadrature_tools.h"
#include "utilities/cu_debug_tools.hcu"

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
    {
      mappings.push_back(new MappingInfo<dim, spacedim>(i));
    }
}

TEST_CASE("Initialize BEMValues on GPU", "[cuda]")
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  FE_Q<dim, spacedim>   fe_for_dirichlet_space(3);
  FE_DGQ<dim, spacedim> fe_for_neumann_space(2);

  const unsigned int                        max_mapping_order = 3;
  std::vector<MappingInfo<dim, spacedim> *> mappings;
  initialize_mappings(mappings, max_mapping_order);

  SauterQuadratureRule<dim> sauter_quad_rule(5, 4, 4, 3);

  // Here we create a dummy mapping support point table which is used to create
  // a @p BEMValues object.
  Table<2, Point<spacedim>>        tria_mapping_support_points;
  BEMValues<dim, spacedim, double> bem_values_cpu(
    fe_for_dirichlet_space,
    fe_for_neumann_space,
    mappings,
    tria_mapping_support_points,
    sauter_quad_rule.quad_rule_for_same_panel,
    sauter_quad_rule.quad_rule_for_common_edge,
    sauter_quad_rule.quad_rule_for_common_vertex,
    sauter_quad_rule.quad_rule_for_regular);

  bem_values_cpu.fill_shape_function_value_tables();

  HierBEM::CUDAWrappers::CUDABEMValues<dim, spacedim, double> bem_values_gpu;
  bem_values_gpu.allocate_and_assign_from_host(bem_values_cpu);

  REQUIRE(is_equal(bem_values_cpu, bem_values_gpu));

  bem_values_gpu.release();
}
