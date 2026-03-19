// Copyright (C) 2006 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file sauter_quadrature_task_producer_buffer.h
 * @brief Definition of the local buffer class for creating Sauter quadrature
 * tasks.
 * @ingroup sauter_quadrature
 *
 * @date 2026-03-14
 * @author Jihuan Tian
 */
#ifndef HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_PRODUCER_BUFFER_H_
#define HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_PRODUCER_BUFFER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>

#include <cstring>
#include <vector>

#include "bem/bem_tools.h"
#include "config.h"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "sauter_quadrature_task_scalar_data.h"
#include "utilities/number_traits.h"

HBEM_NS_OPEN

using namespace dealii;

/**
 * Class for Sauter quadrature task buffer used by a producer to create tasks in
 * batch.
 */
template <int dim, int spacedim, typename RangeNumberType>
class SauterQuadratureTaskProducerBuffer
{
public:
  using real_type = typename numbers::NumberTraits<RangeNumberType>::real_type;

  /**
   * Default constructor
   */
  SauterQuadratureTaskProducerBuffer()
    : capacity(0)
    , task_num(0)
    , cell_neighboring_type(CellNeighboringType::None)
    , scalar_data_buffer(nullptr)
  {}

  /**
   * Allocate memory for the task buffer.
   */
  void
  allocate(const CellNeighboringType neighboring_type,
           const unsigned int        capacity_,
           const unsigned int        max_mapping_n_shape_functions);

  /**
   * Release the allocated memory used by the task buffer.
   */
  void
  release();

  /**
   * Check if the task buffer is full.
   */
  bool
  is_full()
  {
    return task_num == capacity;
  }

  /**
   * Reset the number of tasks to zero without clearing the data in the buffer,
   * since they will be overwritten by new data.
   */
  void
  reset()
  {
    task_num = 0;
  }

  /**
   * Add a Sauter quadrature task to the buffer.
   */
  template <typename KernelNumberType>
  void
  add_task(LAPACKFullMatrixExt<RangeNumberType> *fullmat,
           const unsigned int                    fullmat_row_index,
           const unsigned int                    fullmat_col_index,
           const SauterQuadratureTaskScalarData &scalar_data,
           const real_type                       mass_matrix_entry,
           const PairCellWiseScratchData<dim, spacedim, KernelNumberType>
             &scratch_data);

  /**
   * Capacity of the task buffer.
   */
  unsigned int capacity;
  /**
   * Number of added tasks, which is also the index to add the next task. When
   * <tt>task_num == capacity</tt>, the buffer is full.
   */
  unsigned int task_num;
  /**
   * Cell neighboring type of the tasks added into the buffer.
   *
   * We create a task buffer for each cell neighboring type.
   */
  CellNeighboringType cell_neighboring_type;

  /**
   * List of permuted mapping support points in \f$K_x\f$, which are stored
   * on the host.
   *
   * 1. First dimension: task index in the ring buffer.
   * 2. Second dimension: coordinates of the permuted mapping support points
   * in \f$K_x\f$.
   *
   * The size of the second dimension of this table is allocated with respect
   * to the highest order mapping adopted.
   */
  Table<2, Point<spacedim, real_type>>
    kx_mapping_support_points_permuted_buffer;

  /**
   * List of permuted mapping support points in \f$K_y\f$, which are stored
   * on the host.
   *
   * 1. First dimension: task index in the ring buffer.
   * 2. Second dimension: coordinate of the permuted mapping support points
   * in \f$K_y\f$.
   *
   * The size of the second dimension of this table is allocated with respect
   * to the highest order mapping adopted.
   */
  Table<2, Point<spacedim, real_type>>
    ky_mapping_support_points_permuted_buffer;

  /**
   * Array of pointers to the full matrices held within the near field
   * \hmatrix leaf nodes.
   *
   * \mynote{This array is located on the host, since it is only related to
   * quadrature result assembly.}
   */
  std::vector<LAPACKFullMatrixExt<RangeNumberType> *> fullmat_buffer;

  /**
   * Array of row indices for the entries in the corresponding full matrix.
   *
   * \mynote{This array is located on the host, since it is only related to
   * quadrature result assembly.}
   */
  std::vector<unsigned int> fullmat_row_index_buffer;

  /**
   * Array of column indices for the entries in the corresponding full
   * matrix.
   *
   * \mynote{This array is located on the host, since it is only related to
   * quadrature result assembly.}
   */
  std::vector<unsigned int> fullmat_col_index_buffer;

  /**
   * Array of scalar data collections on the host.
   */
  SauterQuadratureTaskScalarData *scalar_data_buffer;

  /**
   * Mass matrix entries
   */
  std::vector<real_type> mass_matrix_entries_buffer;
};

template <int dim, int spacedim, typename RangeNumberType>
void
SauterQuadratureTaskProducerBuffer<dim, spacedim, RangeNumberType>::allocate(
  const CellNeighboringType neighboring_type,
  const unsigned int        capacity_,
  const unsigned int        max_mapping_n_shape_functions)
{
  task_num              = 0;
  cell_neighboring_type = neighboring_type;
  capacity              = capacity_;

  fullmat_buffer.resize(capacity);
  fullmat_row_index_buffer.resize(capacity);
  fullmat_col_index_buffer.resize(capacity);
  mass_matrix_entries_buffer.resize(capacity);

  kx_mapping_support_points_permuted_buffer.reinit(
    TableIndices<2>(capacity, max_mapping_n_shape_functions));
  ky_mapping_support_points_permuted_buffer.reinit(
    TableIndices<2>(capacity, max_mapping_n_shape_functions));

  if (scalar_data_buffer != nullptr)
    delete[] scalar_data_buffer;
  scalar_data_buffer = new SauterQuadratureTaskScalarData[capacity];
}

template <int dim, int spacedim, typename RangeNumberType>
void
SauterQuadratureTaskProducerBuffer<dim, spacedim, RangeNumberType>::release()
{
  task_num = 0;
  capacity = 0;

  fullmat_buffer.clear();
  fullmat_row_index_buffer.clear();
  fullmat_col_index_buffer.clear();
  mass_matrix_entries_buffer.clear();
  kx_mapping_support_points_permuted_buffer.clear();
  ky_mapping_support_points_permuted_buffer.clear();
  if (scalar_data_buffer != nullptr)
    delete[] scalar_data_buffer;
}

template <int dim, int spacedim, typename RangeNumberType>
template <typename KernelNumberType>
void
SauterQuadratureTaskProducerBuffer<dim, spacedim, RangeNumberType>::add_task(
  LAPACKFullMatrixExt<RangeNumberType> *fullmat,
  const unsigned int                    fullmat_row_index,
  const unsigned int                    fullmat_col_index,
  const SauterQuadratureTaskScalarData &scalar_data,
  const real_type                       mass_matrix_entry,
  const PairCellWiseScratchData<dim, spacedim, KernelNumberType> &scratch_data)
{
  AssertIndexRange(task_num, capacity);

  fullmat_buffer[task_num]             = fullmat;
  fullmat_row_index_buffer[task_num]   = fullmat_row_index;
  fullmat_col_index_buffer[task_num]   = fullmat_col_index;
  mass_matrix_entries_buffer[task_num] = mass_matrix_entry;
  scalar_data_buffer[task_num]         = scalar_data;

  std::memcpy(&kx_mapping_support_points_permuted_buffer(task_num, 0),
              scratch_data.kx_mapping_support_points_permuted.data(),
              sizeof(Point<spacedim, real_type>) *
                scalar_data.kx_mapping_n_shape_functions);
  std::memcpy(&ky_mapping_support_points_permuted_buffer(task_num, 0),
              scratch_data.ky_mapping_support_points_permuted.data(),
              sizeof(Point<spacedim, real_type>) *
                scalar_data.ky_mapping_n_shape_functions);

  task_num++;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_QUADRATURE_SAUTER_QUADRATURE_TASK_PRODUCER_BUFFER_H_
