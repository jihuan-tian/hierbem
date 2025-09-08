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
 * @file cu-qgauss-copy-from-host.cu
 * @brief Verify the creation of @p CUDAQGauss object and copying data from the
 * host object @p QGauss.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-01-30
 */

#include <iostream>

#include "quadrature/cu_qgauss.hcu"

using namespace dealii;
using namespace HierBEM;

int
main()
{
  {
    const unsigned int dim = 1;
    std::cout << "*** dim=" << dim << " ***" << std::endl;

    QGauss<dim>                   quad_cpu(3);
    CUDAWrappers::CUDAQGauss<dim> quad_gpu;

    quad_gpu.allocate(3);
    quad_gpu.assign_from_host(quad_cpu);

    print_cuda_object<<<1, 96>>>(quad_gpu);
    cudaThreadSynchronize();

    quad_gpu.release();
  }

  {
    const unsigned int dim = 2;
    std::cout << "*** dim=" << dim << " ***" << std::endl;

    QGauss<dim>                   quad_cpu(3);
    CUDAWrappers::CUDAQGauss<dim> quad_gpu;

    quad_gpu.allocate(9);
    quad_gpu.assign_from_host(quad_cpu);

    print_cuda_object<<<1, 96>>>(quad_gpu);
    cudaThreadSynchronize();

    quad_gpu.release();
  }

  {
    const unsigned int dim = 3;
    std::cout << "*** dim=" << dim << " ***" << std::endl;

    QGauss<dim>                   quad_cpu(3);
    CUDAWrappers::CUDAQGauss<dim> quad_gpu;

    quad_gpu.allocate(27);
    quad_gpu.assign_from_host(quad_cpu);

    print_cuda_object<<<1, 96>>>(quad_gpu);
    cudaThreadSynchronize();

    quad_gpu.release();
  }

  {
    const unsigned int dim = 4;
    std::cout << "*** dim=" << dim << " ***" << std::endl;

    QGauss<dim>                   quad_cpu(3);
    CUDAWrappers::CUDAQGauss<dim> quad_gpu;

    quad_gpu.allocate(81);
    quad_gpu.assign_from_host(quad_cpu);

    print_cuda_object<<<1, 96>>>(quad_gpu);
    cudaThreadSynchronize();

    quad_gpu.release();
  }

  return 0;
}
