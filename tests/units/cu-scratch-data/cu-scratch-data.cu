/**
 * @file cu-scratch-data.cu
 * @brief Verify @p CUDAPairCellWiseScratchData
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-02-20
 */

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_data.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>

#include <iostream>

#include "bem_values.h"
#include "cu_bem_values.hcu"
#include "debug_tools.hcu"
#include "mapping_q_generic_ext.h"
#include "sauter_quadrature.hcu"
#include "sauter_quadrature_tools.h"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  const unsigned int dim           = 2;
  const unsigned int spacedim      = 3;
  const unsigned     fe_order      = 1;
  const unsigned     mapping_order = 1;

  FE_Q<dim, spacedim> fe_test(fe_order);
  FE_Q<dim, spacedim> fe_trial(fe_order);

  MappingQGenericExt<dim, spacedim> mapping_test(mapping_order);
  MappingQGenericExt<dim, spacedim> mapping_trial(mapping_order);
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    mapping_test_data;
  std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>
    mapping_trial_data;

  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    mapping_test_database =
      mapping_test.get_data(update_default, QGauss<dim>(mapping_order));
  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    mapping_trial_database =
      mapping_trial.get_data(update_default, QGauss<dim>(mapping_order));

  mapping_test_data =
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
      static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
        mapping_test_database.release()));
  mapping_trial_data =
    std::unique_ptr<typename MappingQGeneric<dim, spacedim>::InternalData>(
      static_cast<typename MappingQGeneric<dim, spacedim>::InternalData *>(
        mapping_trial_database.release()));

  SauterQuadratureRule<dim> sauter_quad_rule(5, 4, 4, 3);

  // Compute BEM values.
  BEMValues<dim, spacedim, double> bem_values(
    fe_test,
    fe_trial,
    *mapping_test_data,
    *mapping_trial_data,
    sauter_quad_rule.quad_rule_for_same_panel,
    sauter_quad_rule.quad_rule_for_common_edge,
    sauter_quad_rule.quad_rule_for_common_vertex,
    sauter_quad_rule.quad_rule_for_regular);

  bem_values.fill_shape_function_value_tables();

  // Initialize scratch data.
  PairCellWiseScratchData<dim, spacedim, double> scratch_data(
    fe_test, fe_trial, mapping_test, mapping_trial, bem_values);
  PairCellWisePerTaskData<dim, spacedim, double> copy_data(fe_test, fe_trial);

  HierBEM::CUDAWrappers::CUDAPairCellWiseScratchData<dim, spacedim, double>
                                                     scratch_data_gpu;
  HierBEM::CUDAWrappers::CUDAPairCellWisePerTaskData copy_data_gpu;

  scratch_data_gpu.allocate(scratch_data);
  copy_data_gpu.allocate(copy_data, scratch_data.cuda_stream_handle);

  cudaError_t error_code =
    cudaStreamSynchronize(scratch_data.cuda_stream_handle);
  AssertCuda(error_code);

  scratch_data_gpu.assign_from_host(scratch_data);
  copy_data_gpu.assign_from_host(copy_data, scratch_data.cuda_stream_handle);

  error_code = cudaStreamSynchronize(scratch_data.cuda_stream_handle);
  AssertCuda(error_code);

  scratch_data_gpu.release(scratch_data.cuda_stream_handle);
  copy_data_gpu.release(scratch_data.cuda_stream_handle);

  error_code = cudaStreamSynchronize(scratch_data.cuda_stream_handle);
  AssertCuda(error_code);

  scratch_data.release();
  copy_data.release();

  return 0;
}
