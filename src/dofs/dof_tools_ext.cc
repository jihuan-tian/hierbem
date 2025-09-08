// Copyright (C) 2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

#include "dofs/dof_tools_ext.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

#include <vector>

HBEM_NS_OPEN

using namespace dealii;

namespace DoFToolsExt
{
  void
  generate_full_to_local_dof_id_map(
    const std::vector<bool>              &dof_selectors,
    std::vector<types::global_dof_index> &full_to_local_map)
  {
    AssertDimension(dof_selectors.size(), full_to_local_map.size());

    types::global_dof_index local_i = 0;
    for (types::global_dof_index i = 0; i < dof_selectors.size(); i++)
      {
        if (dof_selectors.at(i))
          {
            full_to_local_map[i] = local_i;
            local_i++;
          }
      }
  }

  void
  generate_local_to_full_dof_id_map(
    const std::vector<bool>              &dof_selectors,
    std::vector<types::global_dof_index> &local_to_full_map)
  {
    AssertDimension(dof_selectors.size(), local_to_full_map.capacity());

    for (types::global_dof_index i = 0; i < dof_selectors.size(); i++)
      {
        if (dof_selectors.at(i))
          local_to_full_map.push_back(i);
      }
  }

  void
  generate_maps_between_full_and_local_dof_ids(
    const std::vector<bool>              &dof_selectors,
    std::vector<types::global_dof_index> &full_to_local_map,
    std::vector<types::global_dof_index> &local_to_full_map)
  {
    AssertDimension(dof_selectors.size(), full_to_local_map.size());
    AssertDimension(dof_selectors.size(), local_to_full_map.capacity());

    types::global_dof_index local_i = 0;
    for (types::global_dof_index i = 0; i < dof_selectors.size(); i++)
      {
        if (dof_selectors.at(i))
          {
            local_to_full_map.push_back(i);
            full_to_local_map[i] = local_i;
            local_i++;
          }
      }
  }
} // namespace DoFToolsExt

HBEM_NS_CLOSE
