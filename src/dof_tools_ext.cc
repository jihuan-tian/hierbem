#include "dof_tools_ext.h"

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