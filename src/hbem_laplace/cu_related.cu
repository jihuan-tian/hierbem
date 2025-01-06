#include <deal.II/base/logstream.h>

#include <cuda_runtime.h>

#include "cu_related.h"
#include "sauter_quadrature.hcu" // for device_properties

using namespace dealii;
using namespace HierBEM;

HBEM_NS_OPEN

void
initCudaRuntime()
{
  const size_t stack_size = 1024 * 10;
  AssertCuda(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
  deallog << "CUDA stack size has been set to " << stack_size << std::endl;

  /**
   * @internal Get GPU device properties.
   */
  AssertCuda(
    cudaGetDeviceProperties(&HierBEM::CUDAWrappers::device_properties, 0));
}

HBEM_NS_CLOSE
