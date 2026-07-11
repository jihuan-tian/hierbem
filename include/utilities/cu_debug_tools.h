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
 * @file cu_debug_tools.h
 * @brief A set of functions for checking consistency of data tables on the host
 * and on the device.
 *
 * @date 2023
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_UTILITIES_CU_DEBUG_TOOLS_H_
#define HIERBEM_INCLUDE_UTILITIES_CU_DEBUG_TOOLS_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "bem/bem_values.h"
#include "bem/cu_bem_values.hcu"
#include "config.h"
#include "linear_algebra/cu_table.hcu"
#include "linear_algebra/cu_table_indices.hcu"
#include "linear_algebra/lapack_full_matrix_ext.h"
#include "quadrature/cu_qgauss.hcu"

HBEM_NS_OPEN

using namespace dealii;

#define AssertCudaThrow(error_code)      \
  AssertThrow(error_code == cudaSuccess, \
              ::ExcCudaError(cudaGetErrorString(error_code)))

/**
 * Check the equality of the quadrature objects on CPU and GPU.
 */
template <int dim>
bool
is_equal(const QGauss<dim>                            &quad_cpu,
         const HierBEM::CUDAWrappers::CUDAQGauss<dim> &quad_gpu)
{
  if (quad_cpu.size() == quad_gpu.size())
    {
      const unsigned int quad_num = quad_cpu.size();
      // Extract quadrature points and weights from the device for comparison.
      Table<2, double> quad_points_from_gpu(quad_num, dim);
      quad_gpu.get_points().copy_to_host(quad_points_from_gpu);

      double *weights_from_gpu = new double[quad_num];
      AssertCuda(cudaMemcpy((void *)weights_from_gpu,
                            (const void *)(quad_gpu.get_weights()),
                            sizeof(double) * quad_num,
                            cudaMemcpyDeviceToHost));

      for (unsigned int q = 0; q < quad_num; q++)
        {
          for (unsigned int i = 0; i < dim; i++)
            {
              if (quad_points_from_gpu(q, i) != quad_cpu.point(q)(i))
                {
                  std::cout << "(" << quad_cpu.point(q)(i) << ","
                            << quad_points_from_gpu(q, i) << ")" << std::endl;
                  delete[] weights_from_gpu;
                  return false;
                }
            }

          if (*(weights_from_gpu + q) != quad_cpu.weight(q))
            {
              std::cout << "(" << quad_cpu.weight(q) << ","
                        << *(weights_from_gpu + q) << ")" << std::endl;
              delete[] weights_from_gpu;
              return false;
            }
        }

      delete[] weights_from_gpu;
      return true;
    }
  else
    {
      std::cout << "(" << quad_cpu.size() << "," << quad_gpu.size() << ")"
                << std::endl;
      return false;
    }
}


/**
 * Check the equality of the vector of values on the CPU and the one dimensional
 * table of values on the GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const std::vector<T>                         &vector_cpu,
         const HierBEM::CUDAWrappers::CUDATable<1, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 1;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (vector_cpu.size() == table_copied_from_gpu.n_elements())
    {
      // Get the pointers to the first element in the two tables.
      const T *vector_cpu_ptr     = vector_cpu.data();
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (std::size_t i = 0; i < vector_cpu.size(); i++)
        {
          if (eps > 0)
            {
              if (std::fabs(*(vector_cpu_ptr + i) - *(table_from_gpu_ptr + i)) >
                  eps)
                {
                  std::cout << "(" << *(vector_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
          else
            {
              if (*(vector_cpu_ptr + i) != *(table_from_gpu_ptr + i))
                {
                  std::cout << "(" << *(vector_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << vector_cpu.size() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the numerical equality of a vector of @p Tensor objects on CPU and a
 * one dimensional table of @p Tensor objects on GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <int spacedim, typename RangeNumberType>
bool
is_equal(const std::vector<Tensor<1, spacedim, RangeNumberType>> &vector_cpu,
         const HierBEM::CUDAWrappers::
           CUDATable<1, Tensor<1, spacedim, RangeNumberType>> &table_gpu,
         const double                                          eps = 0.)
{
  const unsigned int N = 1;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, Tensor<1, spacedim, RangeNumberType>> table_copied_from_gpu(
    table_sizes[0]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (vector_cpu.size() == table_copied_from_gpu.n_elements())
    {
      for (std::size_t i = 0; i < vector_cpu.size(); i++)
        {
          for (unsigned int d = 0; d < spacedim; d++)
            {
              if (eps > 0)
                {
                  if (std::fabs(vector_cpu[i][d] -
                                table_copied_from_gpu(i)[d]) > eps)
                    {
                      std::cout << "(" << vector_cpu[i][d] << ","
                                << table_copied_from_gpu(i)[d] << ")"
                                << std::endl;
                      return false;
                    }
                }
              else
                {
                  if (vector_cpu[i][d] != table_copied_from_gpu(i)[d])
                    {
                      std::cout << "(" << vector_cpu[i][d] << ","
                                << table_copied_from_gpu(i)[d] << ")"
                                << std::endl;
                      return false;
                    }
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << vector_cpu.size() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the numerical equality of a vector of @p Point objects on CPU and a
 * one dimensional table of @p Point objects on GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <int spacedim, typename RangeNumberType>
bool
is_equal(
  const std::vector<Point<spacedim, RangeNumberType>> &vector_cpu,
  const HierBEM::CUDAWrappers::CUDATable<1, Point<spacedim, RangeNumberType>>
              &table_gpu,
  const double eps = 0.)
{
  const unsigned int N = 1;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, Point<spacedim, RangeNumberType>> table_copied_from_gpu(
    table_sizes[0]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (vector_cpu.size() == table_copied_from_gpu.n_elements())
    {
      for (std::size_t i = 0; i < vector_cpu.size(); i++)
        {
          for (unsigned int d = 0; d < spacedim; d++)
            {
              if (eps > 0)
                {
                  if (std::fabs(vector_cpu[i](d) -
                                table_copied_from_gpu(i)(d)) > eps)
                    {
                      std::cout << "(" << vector_cpu[i][d] << ","
                                << table_copied_from_gpu(i)[d] << ")"
                                << std::endl;
                      return false;
                    }
                }
              else
                {
                  if (vector_cpu[i](d) != table_copied_from_gpu(i)(d))
                    {
                      std::cout << "(" << vector_cpu[i][d] << ","
                                << table_copied_from_gpu(i)[d] << ")"
                                << std::endl;
                      return false;
                    }
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << vector_cpu.size() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the equality of the two dimensional table of scalar values on CPU and
 * GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<2, T>                            &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<2, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 2;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0], table_sizes[1]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      // Get the pointers to the first element in the two tables.
      const T *table_cpu_ptr      = &(table_cpu(TableIndices<N>()));
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (std::size_t i = 0; i < table_cpu.n_elements(); i++)
        {
          if (eps > 0)
            {
              if (std::fabs(*(table_cpu_ptr + i) - *(table_from_gpu_ptr + i)) >
                  eps)
                {
                  std::cout << "(" << *(table_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
          else
            {
              if (*(table_cpu_ptr + i) != *(table_from_gpu_ptr + i))
                {
                  std::cout << "(" << *(table_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
          std::cout << "Element #" << i << " is equal" << std::endl;
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the numerical equality of the two dimensional table of rank-1 @p Tensor
 * objects on CPU and on GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <int spacedim, typename RangeNumberType>
bool
is_equal(const Table<2, Tensor<1, spacedim, RangeNumberType>> &table_cpu,
         const HierBEM::CUDAWrappers::
           CUDATable<2, Tensor<1, spacedim, RangeNumberType>> &table_gpu,
         const double                                          eps = 0.)
{
  const unsigned int N = 2;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, Tensor<1, spacedim, RangeNumberType>> table_copied_from_gpu(
    table_sizes[0], table_sizes[1]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      for (std::size_t i = 0; i < table_cpu.n_rows(); i++)
        {
          for (std::size_t j = 0; j < table_cpu.n_cols(); j++)
            {
              for (unsigned int d = 0; d < spacedim; d++)
                {
                  if (eps > 0)
                    {
                      if (std::fabs(table_cpu(i, j)[d] -
                                    table_copied_from_gpu(i, j)[d]) > eps)
                        {
                          std::cout << "(" << table_cpu(i, j)[d] << ","
                                    << table_copied_from_gpu(i, j)[d] << ")"
                                    << std::endl;
                          return false;
                        }
                    }
                  else
                    {
                      if (table_cpu(i, j)[d] != table_copied_from_gpu(i, j)[d])
                        {
                          std::cout << "(" << table_cpu(i, j)[d] << ","
                                    << table_copied_from_gpu(i, j)[d] << ")"
                                    << std::endl;
                          return false;
                        }
                    }
                }
              std::cout << "Element (" << i << "," << j << ") is equal"
                        << std::endl;
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the numerical equality of the three dimensional table of rank-1 @p Tensor
 * objects on CPU and on GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <int spacedim, typename RangeNumberType>
bool
is_equal(const Table<3, Tensor<1, spacedim, RangeNumberType>> &table_cpu,
         const HierBEM::CUDAWrappers::
           CUDATable<3, Tensor<1, spacedim, RangeNumberType>> &table_gpu,
         const double                                          eps = 0.)
{
  const unsigned int N = 3;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, Tensor<1, spacedim, RangeNumberType>> table_copied_from_gpu(
    table_sizes[0], table_sizes[1], table_sizes[2]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      // Get table sizes.
      const unsigned int cell_num     = table_sizes[0];
      const unsigned int fe_shape_num = table_sizes[1];
      const unsigned int quad_num     = table_sizes[2];

      for (std::size_t c = 0; c < cell_num; c++)
        {
          for (std::size_t s = 0; s < fe_shape_num; s++)
            {
              for (std::size_t q = 0; q < quad_num; q++)
                {
                  for (unsigned int d = 0; d < spacedim; d++)
                    {
                      if (eps > 0)
                        {
                          if (std::fabs(table_cpu(c, s, q)[d] -
                                        table_copied_from_gpu(c, s, q)[d]) >
                              eps)
                            {
                              std::cout << "(" << table_cpu(c, s, q)[d] << ","
                                        << table_copied_from_gpu(c, s, q)[d]
                                        << ")" << std::endl;
                              return false;
                            }
                        }
                      else
                        {
                          if (table_cpu(c, s, q)[d] !=
                              table_copied_from_gpu(c, s, q)[d])
                            {
                              std::cout << "(" << table_cpu(c, s, q)[d] << ","
                                        << table_copied_from_gpu(c, s, q)[d]
                                        << ")" << std::endl;
                              return false;
                            }
                        }
                    }
                  std::cout << "Element (" << c << "," << s << "," << q
                            << ") is equal" << std::endl;
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the numerical equality of the two dimensional table of @p Point objects
 * on CPU and on GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <int spacedim, typename RangeNumberType>
bool
is_equal(
  const Table<2, Point<spacedim, RangeNumberType>> &table_cpu,
  const HierBEM::CUDAWrappers::CUDATable<2, Point<spacedim, RangeNumberType>>
              &table_gpu,
  const double eps = 0.)
{
  const unsigned int N = 2;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, Point<spacedim, RangeNumberType>> table_copied_from_gpu(
    table_sizes[0], table_sizes[1]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      for (std::size_t i = 0; i < table_cpu.n_rows(); i++)
        {
          for (std::size_t j = 0; j < table_cpu.n_cols(); j++)
            {
              for (unsigned int d = 0; d < spacedim; d++)
                {
                  if (eps > 0)
                    {
                      if (std::fabs(table_cpu(i, j)[d] -
                                    table_copied_from_gpu(i, j)[d]) > eps)
                        {
                          std::cout << "(" << table_cpu(i, j)[d] << ","
                                    << table_copied_from_gpu(i, j)[d] << ")"
                                    << std::endl;
                          return false;
                        }
                    }
                  else
                    {
                      if (table_cpu(i, j)[d] != table_copied_from_gpu(i, j)[d])
                        {
                          std::cout << "(" << table_cpu(i, j)[d] << ","
                                    << table_copied_from_gpu(i, j)[d] << ")"
                                    << std::endl;
                          return false;
                        }
                    }
                }
              std::cout << "Element (" << i << "," << j << ") is equal"
                        << std::endl;
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the equality of the three dimensional table of scalar values on CPU and
 * GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<3, T>                            &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<3, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 3;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0],
                                    table_sizes[1],
                                    table_sizes[2]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      // Get the pointers to the first element in the two tables.
      const T *table_cpu_ptr      = &(table_cpu(TableIndices<N>()));
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (std::size_t i = 0; i < table_cpu.n_elements(); i++)
        {
          if (eps > 0)
            {
              if (std::fabs(*(table_cpu_ptr + i) - *(table_from_gpu_ptr + i)) >
                  eps)
                {
                  std::cout << "(" << *(table_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
          else
            {
              if (*(table_cpu_ptr + i) != *(table_from_gpu_ptr + i))
                {
                  std::cout << "(" << *(table_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the equality of the four dimensional table of scalar values on CPU and
 * GPU.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<4, T>                            &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<4, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 4;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0],
                                    table_sizes[1],
                                    table_sizes[2],
                                    table_sizes[3]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Check the equality of number of elements.
  if (table_cpu.n_elements() == table_copied_from_gpu.n_elements())
    {
      // Get the pointers to the first element in the two tables.
      const T *table_cpu_ptr      = &(table_cpu(TableIndices<N>()));
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (std::size_t i = 0; i < table_cpu.n_elements(); i++)
        {
          if (eps > 0)
            {
              if (std::fabs(*(table_cpu_ptr + i) - *(table_from_gpu_ptr + i)) >
                  eps)
                {
                  std::cout << "(" << *(table_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
          else
            {
              if (*(table_cpu_ptr + i) != *(table_from_gpu_ptr + i))
                {
                  std::cout << "(" << *(table_cpu_ptr + i) << ","
                            << *(table_from_gpu_ptr + i) << ")" << std::endl;
                  return false;
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the equality of the table of matrices for the gradient of finite
 * element shape functions, applicable to same panel, common edge and common
 * vertex cell neighboring types.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<2, LAPACKFullMatrixExt<T>>       &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<4, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 4;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0],
                                    table_sizes[1],
                                    table_sizes[2],
                                    table_sizes[3]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Get the gradient matrix sizes.
  const unsigned int m = table_cpu(0, 0).m();
  const unsigned int n = table_cpu(0, 0).n();

  // Check the equality of number of elements.
  if (table_cpu.n_elements() * m * n == table_copied_from_gpu.n_elements())
    {
      // Get table sizes.
      const unsigned int k3_terms = table_cpu.n_rows();
      const unsigned int quad_num = table_cpu.n_cols();

      std::size_t counter         = 0;
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));

      for (unsigned int k = 0; k < k3_terms; k++)
        {
          for (unsigned int q = 0; q < quad_num; q++)
            {
              // Here we iterate over matrix entries using the Fortran style
              // of indexing.
              for (unsigned int j = 0; j < n; j++)
                {
                  for (unsigned int i = 0; i < m; i++)
                    {
                      if (eps > 0)
                        {
                          if (std::fabs(*(table_from_gpu_ptr + counter) -
                                        table_cpu(k, q)(i, j)) > eps)
                            {
                              std::cout << "(" << table_cpu(k, q)(i, j) << ","
                                        << *(table_from_gpu_ptr + counter)
                                        << ")" << std::endl;
                              return false;
                            }
                        }
                      else
                        {
                          if (*(table_from_gpu_ptr + counter) !=
                              table_cpu(k, q)(i, j))
                            {
                              std::cout << "(" << table_cpu(k, q)(i, j) << ","
                                        << *(table_from_gpu_ptr + counter)
                                        << ")" << std::endl;
                              return false;
                            }
                        }
                      counter++;
                    }
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() * m * n << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the equality of the table of matrices for the gradient of finite
 * element shape functions, applicable to the regular cell neighboring type.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<1, LAPACKFullMatrixExt<T>>       &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<3, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 3;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());
  Table<N, T> table_copied_from_gpu(table_sizes[0],
                                    table_sizes[1],
                                    table_sizes[2]);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Get the gradient matrix sizes.
  const unsigned int m = table_cpu(0).m();
  const unsigned int n = table_cpu(0).n();

  // Check the equality of number of elements.
  if (table_cpu.n_elements() * m * n == table_copied_from_gpu.n_elements())
    {
      const std::size_t quad_num = table_cpu.n_elements();

      std::size_t counter         = 0;
      const T *table_from_gpu_ptr = &(table_copied_from_gpu(TableIndices<N>()));
      for (unsigned int q = 0; q < quad_num; q++)
        {
          for (unsigned int j = 0; j < n; j++)
            {
              for (unsigned int i = 0; i < m; i++)
                {
                  if (eps > 0)
                    {
                      if (std::fabs(*(table_from_gpu_ptr + counter) -
                                    table_cpu(q)(i, j)) > eps)
                        {
                          std::cout << "(" << table_cpu(q)(i, j) << ","
                                    << *(table_from_gpu_ptr + counter) << ")"
                                    << std::endl;
                          return false;
                        }
                    }
                  else
                    {
                      if (*(table_from_gpu_ptr + counter) != table_cpu(q)(i, j))
                        {
                          std::cout << "(" << table_cpu(q)(i, j) << ","
                                    << *(table_from_gpu_ptr + counter) << ")"
                                    << std::endl;
                          return false;
                        }
                    }
                  counter++;
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "(" << table_cpu.n_elements() * m * n << ","
                << table_copied_from_gpu.n_elements() << ")" << std::endl;
      return false;
    }
}


/**
 * Check the equality of the table of matrices for the gradient of mapping shape
 * functions under different mapping orders, applicable to same panel, common
 * edge and common vertex cell neighboring types.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<3, LAPACKFullMatrixExt<T>>       &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<4, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 4;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());

  const unsigned int mapping_num                      = table_sizes[0];
  const unsigned int k3_terms                         = table_sizes[1];
  const unsigned int quad_num                         = table_sizes[2];
  const unsigned int max_mapping_grad_matrix_elem_num = table_sizes[3];

  Table<N, T> table_copied_from_gpu(mapping_num,
                                    k3_terms,
                                    quad_num,
                                    max_mapping_grad_matrix_elem_num);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Get the number of columns of gradient matrix.
  const unsigned int n = table_cpu(0, 0, 0).n();

  // Check the total number of elements.
  if (table_cpu.n_elements() * max_mapping_grad_matrix_elem_num ==
      table_copied_from_gpu.n_elements())
    {
      for (unsigned int mapping_index = 0; mapping_index < mapping_num;
           mapping_index++)
        {
          // Get the number of rows of the gradient matrix, which depends on the
          // mapping order.
          const unsigned int m = table_cpu(mapping_index, 0, 0).m();

          for (unsigned int k = 0; k < k3_terms; k++)
            {
              for (unsigned int q = 0; q < quad_num; q++)
                {
                  // Here we iterate over matrix entries using the Fortran style
                  // of indexing.
                  for (unsigned int j = 0; j < n; j++)
                    {
                      for (unsigned int i = 0; i < m; i++)
                        {
                          if (eps > 0)
                            {
                              if (std::fabs(
                                    table_copied_from_gpu(
                                      mapping_index, k, q, j * m + i) -
                                    table_cpu(mapping_index, k, q)(i, j)) > eps)
                                {
                                  std::cout
                                    << "("
                                    << table_cpu(mapping_index, k, q)(i, j)
                                    << ","
                                    << table_copied_from_gpu(mapping_index,
                                                             k,
                                                             q,
                                                             j * m + i)
                                    << ")" << std::endl;
                                  return false;
                                }
                            }
                          else
                            {
                              if (table_copied_from_gpu(
                                    mapping_index, k, q, j * m + i) !=
                                  table_cpu(mapping_index, k, q)(i, j))
                                {
                                  std::cout
                                    << "("
                                    << table_cpu(mapping_index, k, q)(i, j)
                                    << ","
                                    << table_copied_from_gpu(mapping_index,
                                                             k,
                                                             q,
                                                             j * m + i)
                                    << ")" << std::endl;
                                  return false;
                                }
                            }
                        }
                    }
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "("
                << table_cpu.n_elements() * max_mapping_grad_matrix_elem_num
                << "," << table_copied_from_gpu.n_elements() << ")"
                << std::endl;
      return false;
    }
}


/**
 * Check the equality of the table of matrices for the gradient of mapping shape
 * functions under different mapping orders, applicable to the regular cell
 * neighboring type.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <typename T>
bool
is_equal(const Table<2, LAPACKFullMatrixExt<T>>       &table_cpu,
         const HierBEM::CUDAWrappers::CUDATable<3, T> &table_gpu,
         const double                                  eps = 0.)
{
  const unsigned int N = 3;

  // Make a copy of the GPU table on the host.
  TableIndices<N> table_sizes;
  HierBEM::CUDAWrappers::copy_table_indices(table_sizes, table_gpu.size());

  const unsigned int mapping_num                      = table_sizes[0];
  const unsigned int quad_num                         = table_sizes[1];
  const unsigned int max_mapping_grad_matrix_elem_num = table_sizes[2];

  Table<N, T> table_copied_from_gpu(mapping_num,
                                    quad_num,
                                    max_mapping_grad_matrix_elem_num);
  table_gpu.copy_to_host(table_copied_from_gpu);

  // Get the number of columns of gradient matrix.
  const unsigned int n = table_cpu(0, 0).n();

  // Check the total number of elements.
  if (table_cpu.n_elements() * max_mapping_grad_matrix_elem_num ==
      table_copied_from_gpu.n_elements())
    {
      for (unsigned int mapping_index = 0; mapping_index < mapping_num;
           mapping_index++)
        {
          // Get the number of rows of the gradient matrix, which depends on the
          // mapping order.
          const unsigned int m = table_cpu(mapping_index, 0).m();

          for (unsigned int q = 0; q < quad_num; q++)
            {
              // Here we iterate over matrix entries using the Fortran style
              // of indexing.
              for (unsigned int j = 0; j < n; j++)
                {
                  for (unsigned int i = 0; i < m; i++)
                    {
                      if (eps > 0)
                        {
                          if (std::fabs(table_copied_from_gpu(mapping_index,
                                                              q,
                                                              j * m + i) -
                                        table_cpu(mapping_index, q)(i, j)) >
                              eps)
                            {
                              std::cout << "("
                                        << table_cpu(mapping_index, q)(i, j)
                                        << ","
                                        << table_copied_from_gpu(mapping_index,
                                                                 q,
                                                                 j * m + i)
                                        << ")" << std::endl;
                              return false;
                            }
                        }
                      else
                        {
                          if (table_copied_from_gpu(mapping_index,
                                                    q,
                                                    j * m + i) !=
                              table_cpu(mapping_index, q)(i, j))
                            {
                              std::cout << "("
                                        << table_cpu(mapping_index, q)(i, j)
                                        << ","
                                        << table_copied_from_gpu(mapping_index,
                                                                 q,
                                                                 j * m + i)
                                        << ")" << std::endl;
                              return false;
                            }
                        }
                    }
                }
            }
        }
      return true;
    }
  else
    {
      std::cout << "("
                << table_cpu.n_elements() * max_mapping_grad_matrix_elem_num
                << "," << table_copied_from_gpu.n_elements() << ")"
                << std::endl;
      return false;
    }
}


/**
 * Check the equality of @p BEMValues and @p CUDABEMValues.
 *
 * @param eps When eps > 0, perform tolerance comparison; when eps <= 0, perform
 * exact comparison.
 */
template <int dim, int spacedim, typename RangeNumberType = double>
bool
is_equal(
  const BEMValues<dim, spacedim, RangeNumberType> &bem_values_cpu,
  const HierBEM::CUDAWrappers::CUDABEMValues<dim, spacedim, RangeNumberType>
                               &bem_values_gpu,
  [[maybe_unused]] const double eps = 0.)
{
  if (bem_values_cpu.is_surface_curl_needed ==
      bem_values_gpu.is_surface_curl_needed)
    {
      std::cout << "is_surface_curl_needed is equal" << std::endl;
    }
  else
    {
      std::cout << "is_surface_curl_needed is not equal" << std::endl;
      return false;
    }

  // Check quadrature rules.
  if (is_equal(bem_values_cpu.quad_rule_for_same_panel,
               bem_values_gpu.quad_rule_for_same_panel))
    {
      std::cout << "quad_rule_for_same_panel is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_same_panel is not equal" << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.quad_rule_for_common_edge,
               bem_values_gpu.quad_rule_for_common_edge))
    {
      std::cout << "quad_rule_for_common_edge is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_common_edge is not equal" << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.quad_rule_for_common_vertex,
               bem_values_gpu.quad_rule_for_common_vertex))
    {
      std::cout << "quad_rule_for_common_vertex is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_common_vertex is not equal" << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.quad_rule_for_regular,
               bem_values_gpu.quad_rule_for_regular))
    {
      std::cout << "quad_rule_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_rule_for_regular is not equal" << std::endl;
      return false;
    }

  // Check finite element shape function value tables.
  if (is_equal(bem_values_cpu.kx_shape_value_table_for_same_panel,
               bem_values_gpu.kx_shape_value_table_for_same_panel))
    {
      std::cout << "kx_shape_value_table_for_same_panel is equal" << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_same_panel is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_value_table_for_same_panel,
               bem_values_gpu.ky_shape_value_table_for_same_panel))
    {
      std::cout << "ky_shape_value_table_for_same_panel is equal" << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_same_panel is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_shape_value_table_for_common_edge,
               bem_values_gpu.kx_shape_value_table_for_common_edge))
    {
      std::cout << "kx_shape_value_table_for_common_edge is equal" << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_common_edge is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_value_table_for_common_edge,
               bem_values_gpu.ky_shape_value_table_for_common_edge))
    {
      std::cout << "ky_shape_value_table_for_common_edge is equal" << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_common_edge is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_shape_value_table_for_common_vertex,
               bem_values_gpu.kx_shape_value_table_for_common_vertex))
    {
      std::cout << "kx_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_common_vertex is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_value_table_for_common_vertex,
               bem_values_gpu.ky_shape_value_table_for_common_vertex))
    {
      std::cout << "ky_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_common_vertex is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_shape_value_table_for_regular,
               bem_values_gpu.kx_shape_value_table_for_regular))
    {
      std::cout << "kx_shape_value_table_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "kx_shape_value_table_for_regular is not equal" << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_value_table_for_regular,
               bem_values_gpu.ky_shape_value_table_for_regular))
    {
      std::cout << "ky_shape_value_table_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "ky_shape_value_table_for_regular is not equal" << std::endl;
      return false;
    }

  // Check mapping shape function value tables.
  if (is_equal(bem_values_cpu.kx_mapping_shape_value_table_for_same_panel,
               bem_values_gpu.kx_mapping_shape_value_table_for_same_panel))
    {
      std::cout << "kx_mapping_shape_value_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_same_panel is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_mapping_shape_value_table_for_same_panel,
               bem_values_gpu.ky_mapping_shape_value_table_for_same_panel))
    {
      std::cout << "ky_mapping_shape_value_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_same_panel is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_mapping_shape_value_table_for_common_edge,
               bem_values_gpu.kx_mapping_shape_value_table_for_common_edge))
    {
      std::cout << "kx_mapping_shape_value_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_common_edge is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_mapping_shape_value_table_for_common_edge,
               bem_values_gpu.ky_mapping_shape_value_table_for_common_edge))
    {
      std::cout << "ky_mapping_shape_value_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_common_edge is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_mapping_shape_value_table_for_common_vertex,
               bem_values_gpu.kx_mapping_shape_value_table_for_common_vertex))
    {
      std::cout << "kx_mapping_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_mapping_shape_value_table_for_common_vertex is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_mapping_shape_value_table_for_common_vertex,
               bem_values_gpu.ky_mapping_shape_value_table_for_common_vertex))
    {
      std::cout << "ky_mapping_shape_value_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_mapping_shape_value_table_for_common_vertex is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.mapping_shape_value_table_for_regular,
               bem_values_gpu.mapping_shape_value_table_for_regular))
    {
      std::cout << "mapping_shape_value_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "mapping_shape_value_table_for_regular is not equal"
                << std::endl;
      return false;
    }

  // Check finite element shape function gradient tables.
  if (is_equal(bem_values_cpu.kx_shape_grad_matrix_table_for_same_panel,
               bem_values_gpu.kx_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "kx_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_same_panel is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_grad_matrix_table_for_same_panel,
               bem_values_gpu.ky_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "ky_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_same_panel is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_shape_grad_matrix_table_for_common_edge,
               bem_values_gpu.kx_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_edge is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_grad_matrix_table_for_common_edge,
               bem_values_gpu.ky_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_edge is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_shape_grad_matrix_table_for_common_vertex,
               bem_values_gpu.kx_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_common_vertex is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_grad_matrix_table_for_common_vertex,
               bem_values_gpu.ky_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_vertex is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_common_vertex is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.kx_shape_grad_matrix_table_for_regular,
               bem_values_gpu.kx_shape_grad_matrix_table_for_regular))
    {
      std::cout << "kx_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "kx_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.ky_shape_grad_matrix_table_for_regular,
               bem_values_gpu.ky_shape_grad_matrix_table_for_regular))
    {
      std::cout << "ky_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "ky_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
      return false;
    }

  // Check mapping shape function gradient tables.
  if (is_equal(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_same_panel,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "kx_mapping_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_same_panel is not equal"
        << std::endl;
      return false;
    }

  if (is_equal(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_same_panel,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_same_panel))
    {
      std::cout << "ky_mapping_shape_grad_matrix_table_for_same_panel is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_same_panel is not equal"
        << std::endl;
      return false;
    }

  if (is_equal(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_common_edge,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "kx_mapping_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_common_edge is not equal"
        << std::endl;
      return false;
    }

  if (is_equal(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_common_edge,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_common_edge))
    {
      std::cout << "ky_mapping_shape_grad_matrix_table_for_common_edge is equal"
                << std::endl;
    }
  else
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_common_edge is not equal"
        << std::endl;
      return false;
    }

  if (is_equal(
        bem_values_cpu.kx_mapping_shape_grad_matrix_table_for_common_vertex,
        bem_values_gpu.kx_mapping_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_common_vertex is equal"
        << std::endl;
    }
  else
    {
      std::cout
        << "kx_mapping_shape_grad_matrix_table_for_common_vertex is not equal"
        << std::endl;
      return false;
    }

  if (is_equal(
        bem_values_cpu.ky_mapping_shape_grad_matrix_table_for_common_vertex,
        bem_values_gpu.ky_mapping_shape_grad_matrix_table_for_common_vertex))
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_common_vertex is equal"
        << std::endl;
    }
  else
    {
      std::cout
        << "ky_mapping_shape_grad_matrix_table_for_common_vertex is not equal"
        << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.mapping_shape_grad_matrix_table_for_regular,
               bem_values_gpu.mapping_shape_grad_matrix_table_for_regular))
    {
      std::cout << "mapping_shape_grad_matrix_table_for_regular is equal"
                << std::endl;
    }
  else
    {
      std::cout << "mapping_shape_grad_matrix_table_for_regular is not equal"
                << std::endl;
      return false;
    }

  // Compare precomputed cell data at quadrature points.
  if (is_equal(bem_values_cpu.JxW_at_quad_points_for_regular,
               bem_values_gpu.JxW_at_quad_points_for_regular))
    {
      std::cout << "JxW_at_quad_points_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "JxW_at_quad_points_for_regular is not equal" << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.normals_at_quad_points_for_regular,
               bem_values_gpu.normals_at_quad_points_for_regular))
    {
      std::cout << "normals_at_quad_points_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "normals_at_quad_points_for_regular is not equal"
                << std::endl;
      return false;
    }

  if (is_equal(bem_values_cpu.quad_points_for_regular,
               bem_values_gpu.quad_points_for_regular))
    {
      std::cout << "quad_points_for_regular is equal" << std::endl;
    }
  else
    {
      std::cout << "quad_points_for_regular is not equal" << std::endl;
      return false;
    }

  if (bem_values_cpu.is_surface_curl_needed)
    {
      if (is_equal(bem_values_cpu.kx_shape_curls_at_quad_points_for_regular,
                   bem_values_gpu.kx_shape_curls_at_quad_points_for_regular))
        {
          std::cout << "kx_shape_curls_at_quad_points_for_regular is equal"
                    << std::endl;
        }
      else
        {
          std::cout << "kx_shape_curls_at_quad_points_for_regular is not equal"
                    << std::endl;
          return false;
        }

      if (is_equal(bem_values_cpu.ky_shape_curls_at_quad_points_for_regular,
                   bem_values_gpu.ky_shape_curls_at_quad_points_for_regular))
        {
          std::cout << "ky_shape_curls_at_quad_points_for_regular is equal"
                    << std::endl;
        }
      else
        {
          std::cout << "ky_shape_curls_at_quad_points_for_regular is not equal"
                    << std::endl;
          return false;
        }
    }

  return true;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_UTILITIES_CU_DEBUG_TOOLS_H_
