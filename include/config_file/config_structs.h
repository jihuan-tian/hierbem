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
 * @file config_structs.h
 * @brief Declaration of structs used for program configuration.
 *
 * @date 2026-02-04
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_STRUCTS_H_
#define HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_STRUCTS_H_

#include <complex>
#include <cstdint>

#include "bem/types.h"
#include "config.h"
#include "quadrature/sauter_quadrature_tools.h"

HBEM_NS_OPEN

/**
 * Configurations for Laplace BEM solver.
 */
struct ConfLaplaceBEM
{
  std::uint32_t fe_order_for_dirichlet_space =
    1;                                          // FE order for Dirichlet space
  std::uint32_t fe_order_for_neumann_space = 0; // FE order for Neumann space
  ProblemType   problem_type;
  bool          is_interior_problem = false; // Whether the problem is interior
};

/**
 * Configurations for Helmholtz acoustic solver.
 */
struct ConfHelmholtzAcousticBEM
{
  std::complex<double> kappa; // Wave number
  std::uint32_t        fe_order_for_dirichlet_space =
    1;                                          // FE order for Dirichlet space
  std::uint32_t fe_order_for_neumann_space = 0; // FE order for Neumann space
  ProblemType   problem_type;
  bool          is_interior_problem = false; // Whether the problem is interior
};

/**
 * Configuration for H-matrix construction
 */
struct ConfHMatrix
{
  std::uint32_t n_min_for_ct  = 64;  // n_min for ClusterTree
  std::uint32_t n_min_for_bct = 64;  // n_min for BlockClusterTree
  double        eta           = 0.8; // eta for H-matrix
  std::uint32_t max_rank      = 100; // max_rank for H-matrix
  double        aca_relative_err =
    0.001; // for ACA: max relative error while assembling H-matrix
  // Assemble H-matrix using the old implementation, which is serial on the host
  // and does not use the producer-consumer model on the device. This method is
  // obsolete and is only used for test and performance profile.
  bool cpu_serial_without_producer_consumer = false;
};

struct ConfSauterQuadNearField
{
  // Number of producer threads.
  std::uint32_t producer_num = 2;
  // Capacity of producer's local task buffers for various cell neighboring
  // types.
  std::uint32_t producer_buffer_capacity = 4096;
  // Number of consumer threads for various cell neighboring types.
  std::uint32_t consumer_num_same_panel    = 1;
  std::uint32_t consumer_num_common_edge   = 1;
  std::uint32_t consumer_num_common_vertex = 1;
  std::uint32_t consumer_num_regular       = 2;
  // Number of Sauter quadrature tasks in a thread block.
  std::uint32_t task_num_per_block = 256;
  // Number of thread blocks in a task batch.
  std::uint32_t block_num_per_batch = 8;
  // Number of task batches in a ring buffer, which determines the capacity of
  // the buffer.
  std::uint32_t batch_num_in_buffer = 8;
  // Wait time in microseconds for the task processing loop in a consumer
  // thread.
  std::uint32_t consumer_wait_time_us = 1;
};

struct ConfSauterQuadFarField
{
  // Number of vector entries can be held by a task buffer, which is used to
  // determine the buffer's capacity.
  std::uint32_t vector_entry_num_in_task_buffer = 10000;
  // Number of Sauter quadrature tasks in a thread block, which is in the x
  // direction of the thread grid.
  std::uint32_t task_num_per_block = 128;
  // Number of quadrature points in a thread block, which is in the y direction
  // of the thread grid.
  std::uint32_t quad_point_num_per_block = 4;
  // Number of threads per block, which is also the number of vector entries,
  // when accumulating quadrature results.
  std::uint32_t vector_entry_num_per_block = 128;
};

/**
 * Configuration for Sauter quadrature
 */
struct ConfSauterQuad
{
  ConfSauterQuadNearField near_field;
  ConfSauterQuadFarField  far_field;
  // Quad order for the single layer potential operator
  SauterQuadOrder slp_order{5, 4, 4, 3};
  // Quad order for the double layer potential operator
  SauterQuadOrder dlp_order{5, 4, 4, 3};
  // Quad order for the adjoint double layer potential operator
  SauterQuadOrder adlp_order{5, 4, 4, 3};
  // Quad order for the hyper singular operator
  SauterQuadOrder hyper_singular_order{5, 4, 4, 3};
};

/**
 * Configurations for linear system solvers
 */
struct ConfLinearSolver
{
  std::uint32_t max_iter    = 1000; // Maximum iterations
  double        abs_tol     = 1e-8; // Absolute tolerance
  bool          log_history = true; // If logging the iteration history
  bool log_result = true; // If logging the start and end iteration steps
};

/**
 * Configurations for operator preconditioner
 */
struct ConfOperatorPreconditioner
{
  std::uint32_t max_iter = 1000; // Maximum iterations
  double        abs_tol  = 1e-8; // Absolute tolerance
  double        omega =
    1.0; // Relaxation factor for the internal Jacobi preconditioner
  bool log_history = true; // If logging the iteration history
  bool log_result  = true; // If logging the start and end iteration steps
};

/**
 * Parallelization configurations
 */
struct ConfParallelization
{
  std::int32_t tbb_thread_num =
    -1; // number of threads used by TBB, -1 means using all available threads
  std::uint32_t cuda_stack_size_kb = 10240; // CUDA device stack size in KB
  // Grain size for the TBB blocked range of H-matrix far field leaf nodes, used
  // for building these nodes.
  std::uint32_t hmat_far_field_grain_size = 1;
  // Grain size for the TBB blocked range of H-matrix near field leaf nodes,
  // used for adding stabilization terms when the boundary integral operator is
  // hyper singular.
  std::uint32_t hmat_near_field_stabilization_grain_size = 1;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_CONFIG_FILE_CONFIG_STRUCTS_H_
