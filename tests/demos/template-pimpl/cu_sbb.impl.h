#ifndef CU_SBB_IMPL_H__
#define CU_SBB_IMPL_H__

#include <cuda_runtime.h>

#include "cu_sbb.h"

template <typename NumType>
__host__ __device__ NumType
add_(NumType a, NumType b)
{
  return a + b;
}

template <typename NumType>
__global__ void
vec_add_(NumType *a, NumType *b, NumType *c, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = add_<NumType>(a[i], b[i]);
}

template <typename NumType>
void
CuSbb<NumType>::vec_add(NumType *a, NumType *b, NumType *c, int n)
{
  NumType *dev_a, *dev_b, *dev_c;

  cudaMallocManaged(&dev_a, n * sizeof(NumType));
  cudaMallocManaged(&dev_b, n * sizeof(NumType));
  cudaMallocManaged(&dev_c, n * sizeof(NumType));

  cudaMemcpy(dev_a, a, n * sizeof(NumType), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, n * sizeof(NumType), cudaMemcpyHostToDevice);

  vec_add_<<<(n + 255) / 256, 256>>>(dev_a, dev_b, dev_c, n);

  cudaMemcpy(c, dev_c, n * sizeof(NumType), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

#endif // CU_SBB_IMPL_H__