// Copyright (C) 2024-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#ifndef HIERBEM_INCLUDE_DOFS_DOF_TO_CELL_TOPOLOGY_H_
#define HIERBEM_INCLUDE_DOFS_DOF_TO_CELL_TOPOLOGY_H_

#include <deal.II/dofs/dof_handler.h>

#include "config.h"

HBEM_NS_OPEN

using namespace dealii;

template <int dim, int spacedim>
class DoFToCellTopology
{
public:
  /**
   * The core structure describing the DoF-to-cell topology. The index for the
   * outer vector is a full and external DoF index, i.e. the original DoF
   * indices defined in a DoF handler. Each element in the outer vector is
   * another vector of cell iterator pointers. The related cells share this
   * common DoF.
   */
  std::vector<
    std::vector<const typename DoFHandler<dim, spacedim>::cell_iterator *>>
    topology;

  /**
   * Maximum number of cells associated with a DoF.
   */
  unsigned int max_cells_per_dof;
};

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_DOFS_DOF_TO_CELL_TOPOLOGY_H_
