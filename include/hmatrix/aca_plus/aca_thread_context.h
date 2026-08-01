// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file aca_thread_context.h
 * @brief Define a TBB thread context class for holding thread local data.
 * @ingroup hierarchical_matrices
 *
 * @date 2026-07-31
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_THREAD_CONTEXT_H_
#define HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_THREAD_CONTEXT_H_

#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>

#include <vector>

#include "bem/bem_values.h"
#include "config.h"
#include "mapping/mapping_info.h"
#include "quadrature/sauter_quadrature_task_buffer_for_vector.hcu"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

template <int dim,
          int spacedim,
          typename RangeNumberType,
          typename KernelNumberType>
struct ACAThreadContext
{
  using real_type = typename numbers::NumberTraits<KernelNumberType>::real_type;

  ACAThreadContext(const FiniteElement<dim, spacedim> &kx_fe,
                   const FiniteElement<dim, spacedim> &ky_fe,
                   const unsigned int                  n_max_vector_entries,
                   const unsigned int max_cells_per_dof_in_test_space,
                   const unsigned int max_cells_per_dof_in_ansatz_space,
                   const std::vector<MappingInfo<dim, spacedim> *> &mappings,
                   const BEMValues<dim, spacedim, real_type>       &bem_values)
    : scratch_data(kx_fe, ky_fe, mappings)
    , copy_data(kx_fe, ky_fe)
    , sauter_task_buffer_for_vector(n_max_vector_entries,
                                    max_cells_per_dof_in_test_space,
                                    max_cells_per_dof_in_ansatz_space,
                                    bem_values,
                                    scratch_data.cuda_stream_handle)
  {}

  ACAThreadContext(const ACAThreadContext &) = delete;

  ACAThreadContext(ACAThreadContext &&) = delete;

  ACAThreadContext &
  operator=(const ACAThreadContext &) = delete;

  ACAThreadContext &
  operator=(ACAThreadContext &&) = delete;

  PairCellWiseScratchDataForHMatrixFarField<dim, spacedim, KernelNumberType>
                                         scratch_data;
  PairCellWisePerTaskData<dim, spacedim> copy_data;
  HierBEM::CUDAWrappers::SauterQuadratureTaskBufferForVector<
    dim,
    spacedim,
    DeviceNumberType<RangeNumberType>,
    DeviceNumberType<KernelNumberType>>
    sauter_task_buffer_for_vector;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_ACA_PLUS_ACA_THREAD_CONTEXT_H_
