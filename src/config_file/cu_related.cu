// Copyright (C) 2025 Xiaozhe Wang <chaoslawful@gmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#include "config.h"
#include "config_file/cu_related.h"
#include "quadrature/cu_sauter_quadrature.hcu" // for device_properties

using namespace dealii;
using namespace HierBEM;

HBEM_NS_OPEN

void
initCudaRuntime(const ConfParallelization &parallel_params)
{
  std::uint32_t stack_size = parallel_params.cuda_stack_size_kb;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));
}

HBEM_NS_CLOSE
