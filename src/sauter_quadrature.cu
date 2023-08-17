/**
 * @file sauter_quadrature.cpp
 * @brief Introduction of sauter_quadrature.cpp
 *
 * @date 2022-03-03
 * @author Jihuan Tian
 */

#include "sauter_quadrature.hcu"

namespace HierBEM
{
  namespace CUDAWrappers
  {
    /**
     * Global variable storing the properties of the GPU device. The information
     * will be use for automatically determining the thread block size and block
     * rectangle for parallelization on the GPU device.
     */
    cudaDeviceProp device_properties;
  } // namespace CUDAWrappers

  void
  print_dof_to_cell_topology(
    const std::vector<std::vector<unsigned int>> &dof_to_cell_topo)
  {
    unsigned int counter = 0;
    for (const auto &dof_to_cell : dof_to_cell_topo)
      {
        std::cout << counter << ": ";
        for (auto cell_index : dof_to_cell)
          {
            std::cout << cell_index << " ";
          }
        std::cout << std::endl;
        counter++;
      }
  }
} // namespace HierBEM
