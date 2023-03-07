/**
 * @file cu-bem-values.cu
 * @brief Verify the initialization of BEMValues on GPU.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-01
 */

#include <iostream>

#include "cu_bem_values.hcu"
#include "laplace_bem.h"

using namespace dealii;
using namespace IdeoBEM;

#include "debug_tools.hcu"

int
main()
{
  const unsigned     dim      = 2;
  const unsigned int spacedim = 3;

  FE_Q<dim, spacedim>   fe_for_dirichlet_space(3);
  FE_DGQ<dim, spacedim> fe_for_neumann_space(2);

  MappingQGenericExt<dim, spacedim> kx_mapping(1);
  MappingQGenericExt<dim, spacedim> ky_mapping(1);

  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    kx_mapping_data;
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    ky_mapping_data;

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    kx_mapping_database = kx_mapping.get_data(update_default, QGauss<dim>(1));

  /**
   * Downcast the smart pointer of @p Mapping<dim, spacedim>::InternalDataBase to
   * @p MappingQGeneric<dim,spacedim>::InternalData by first unwrapping
   * the original smart pointer via @p static_cast then wrapping it again.
   */
  kx_mapping_data =
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
      static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
        kx_mapping_database.release()));

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    ky_mapping_database = ky_mapping.get_data(update_default, QGauss<dim>(1));

  ky_mapping_data =
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
      static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
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

  IdeoBEM::CUDAWrappers::CUDABEMValues<dim, spacedim> bem_values_gpu;
  bem_values_gpu.allocate_and_assign_from_host(bem_values_cpu);

  Assert(is_equal(bem_values_cpu, bem_values_gpu), ExcInternalError());

  bem_values_gpu.release();

  return 0;
}
