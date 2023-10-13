/**
 * @file verify-cuda-table.cu
 * @brief Verify the implementation of @p CUDATable class, whose memory is
 * managed from the host.
 *
 * @ingroup testers
 * @author Jihuan Tian
 * @date 2023-01-16
 */

#include <iostream>
#include <vector>

#include "cu_table.hcu"


using namespace HierBEM;
using namespace dealii;

int
main()
{
  {
    std::cout << "=== Synchronous allocation and releasing" << std::endl;

    std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8};

    TableIndices<3>  indices(2, 2, 2);
    Table<3, double> table_cpu(2, 2, 2, values.begin());

    CUDAWrappers::CUDATable<3, double> table_gpu1;
    CUDAWrappers::CUDATable<3, double> table_gpu2;
    table_gpu1.allocate(indices);
    table_gpu1.assign_from_host(table_cpu);
    std::cout << "Table1:" << std::endl;
    print_cuda_object<<<1, 10>>>(table_gpu1);
    cudaThreadSynchronize();

    table_gpu2 = table_gpu1;
    std::cout << "Table2:" << std::endl;
    print_cuda_object<<<1, 10>>>(table_gpu2);
    cudaThreadSynchronize();

    table_gpu1.release();
    table_gpu2.release();
  }

  {
    std::cout << "=== Asynchronous allocation and releasing" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<double> values{1, 2, 3, 4, 5, 6, 7, 8};

    TableIndices<3>  indices(2, 2, 2);
    Table<3, double> table_cpu(2, 2, 2, values.begin());

    CUDAWrappers::CUDATable<3, double> table_gpu1;
    CUDAWrappers::CUDATable<3, double> table_gpu2;

    table_gpu1.allocate(indices, stream);
    table_gpu1.assign_from_host(table_cpu, stream);
    cudaStreamSynchronize(stream);

    std::cout << "Table1:" << std::endl;
    print_cuda_object<<<1, 10>>>(table_gpu1);
    cudaThreadSynchronize();

    table_gpu2.assign(table_gpu1, stream);
    cudaStreamSynchronize(stream);
    std::cout << "Table2:" << std::endl;
    print_cuda_object<<<1, 10>>>(table_gpu2);
    cudaThreadSynchronize();

    table_gpu1.release(stream);
    table_gpu2.release(stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
  }

  return 0;
}
