// Copyright (C) 2022-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file sauter_quadrature.cpp
 * @brief Introduction of sauter_quadrature.cpp
 *
 * @date 2022-03-03
 * @author Jihuan Tian
 */

#include "config.h"
#include "quadrature/sauter_quadrature.hcu"

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
