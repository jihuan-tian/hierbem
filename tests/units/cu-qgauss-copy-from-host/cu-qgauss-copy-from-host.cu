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
