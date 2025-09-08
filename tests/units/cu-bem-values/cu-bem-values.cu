// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
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

#include <iostream>

#include "bem/cu_bem_values.hcu"
#include "laplace/laplace_bem.h"

using namespace dealii;
using namespace HierBEM;

#include "debug_tools.hcu"

int
main()
{
  const unsigned     dim      = 2;
  const unsigned int spacedim = 3;

  FE_Q<dim, spacedim>   fe_for_dirichlet_space(3);
  FE_DGQ<dim, spacedim> fe_for_neumann_space(2);

  MappingQExt<dim, spacedim> kx_mapping(1);
  MappingQExt<dim, spacedim> ky_mapping(1);

  std::unique_ptr<typename MappingQ<dim, spacedim>::InternalData>
    kx_mapping_data;
  std::unique_ptr<typename MappingQ<dim, spacedim>::InternalData>
    ky_mapping_data;

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    kx_mapping_database = kx_mapping.get_data(update_default, QGauss<dim>(1));

  /**
   * Downcast the smart pointer of @p Mapping<dim, spacedim>::InternalDataBase to
   * @p MappingQ<dim,spacedim>::InternalData by first unwrapping
   * the original smart pointer via @p static_cast then wrapping it again.
   */
  kx_mapping_data =
    std::unique_ptr<typename MappingQ<dim, spacedim>::InternalData>(
      static_cast<typename MappingQ<dim, spacedim>::InternalData *>(
        kx_mapping_database.release()));

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    ky_mapping_database = ky_mapping.get_data(update_default, QGauss<dim>(1));

  ky_mapping_data =
    std::unique_ptr<typename MappingQ<dim, spacedim>::InternalData>(
      static_cast<typename MappingQ<dim, spacedim>::InternalData *>(
        ky_mapping_database.release()));

  SauterQuadratureRule<dim> sauter_quad_rule(5, 4, 4, 3);

  BEMValues<dim, spacedim, double> bem_values_cpu(
    fe_for_dirichlet_space,
    fe_for_neumann_space,
    *kx_mapping_data,
    *ky_mapping_data,
    sauter_quad_rule.quad_rule_for_same_panel,
    sauter_quad_rule.quad_rule_for_common_edge,
    sauter_quad_rule.quad_rule_for_common_vertex,
    sauter_quad_rule.quad_rule_for_regular);

  bem_values_cpu.fill_shape_function_value_tables();

  HierBEM::CUDAWrappers::CUDABEMValues<dim, spacedim> bem_values_gpu;
  bem_values_gpu.allocate_and_assign_from_host(bem_values_cpu);

  Assert(is_equal(bem_values_cpu, bem_values_gpu), ExcInternalError());

  bem_values_gpu.release();

  return 0;
}
