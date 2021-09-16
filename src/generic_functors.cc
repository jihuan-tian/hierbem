/**
 * \file generic_functors.cc
 * \brief Introduction of generic_functors.cc
 * \date 2021-07-31
 * \author Jihuan Tian
 */

#include "generic_functors.h"

void
build_index_set_global_to_local_map(
  const std::vector<dealii::types::global_dof_index>
    &index_set_as_local_to_global_map,
  std::map<dealii::types::global_dof_index, size_t> &global_to_local_map)
{
  global_to_local_map.clear();

  for (size_t i = 0; i < index_set_as_local_to_global_map.size(); i++)
    {
      global_to_local_map.emplace(
        std::make_pair(index_set_as_local_to_global_map[i], i));
    }
}
