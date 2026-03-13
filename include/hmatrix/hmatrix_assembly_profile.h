// Copyright (C) 2026 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file hmatrix_assembly_profile.h
 * @brief Struct containing the profile data for H-matrix assembly.
 *
 * @date 2026-03-12
 * @author Jihuan Tian
 */

#ifndef HIERBEM_INCLUDE_HMATRIX_HMATRIX_ASSEMBLY_PROFILE_H_
#define HIERBEM_INCLUDE_HMATRIX_HMATRIX_ASSEMBLY_PROFILE_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>

#include "config.h"

HBEM_NS_OPEN

struct HMatrixAssemblyProfile
{
  friend std::ostream &
  operator<<(std::ostream &out, const HMatrixAssemblyProfile &profile);

  HMatrixAssemblyProfile()
    : near_field_total_time(0.0)
    , far_field_total_time(0.0)
    , producer_time_per_type{}
    , consumer_time_per_type{}
    , near_field_tasks_per_type{}
  {
    for (int i = 0; i < 4; i++)
      {
        producer_time_per_type[i].store(0.0);
        consumer_time_per_type[i].store(0.0);
        near_field_tasks_per_type[i].store(0);
      }
  }

  double                                    near_field_total_time;
  double                                    far_field_total_time;
  std::array<std::atomic<double>, 4>        producer_time_per_type;
  std::array<std::atomic<double>, 4>        consumer_time_per_type;
  std::array<std::atomic<std::uint64_t>, 4> near_field_tasks_per_type;
};

inline std::ostream &
operator<<(std::ostream &out, const HMatrixAssemblyProfile &profile)
{
  out << "near_field_total_time=" << profile.near_field_total_time << "\n"
      << "far_field_total_time=" << profile.far_field_total_time << "\n"
      << "producer_time_for_same_panel="
      << profile.producer_time_per_type[0].load() << "\n"
      << "producer_time_for_common_edge="
      << profile.producer_time_per_type[1].load() << "\n"
      << "producer_time_for_common_vertex="
      << profile.producer_time_per_type[2].load() << "\n"
      << "producer_time_for_regular="
      << profile.producer_time_per_type[3].load() << "\n"
      << "consumer_time_for_same_panel="
      << profile.consumer_time_per_type[0].load() << "\n"
      << "consumer_time_for_common_edge="
      << profile.consumer_time_per_type[1].load() << "\n"
      << "consumer_time_for_common_vertex="
      << profile.consumer_time_per_type[2].load() << "\n"
      << "consumer_time_for_regular="
      << profile.consumer_time_per_type[3].load() << "\n"
      << "near_field_tasks_for_same_panel="
      << profile.near_field_tasks_per_type[0].load() << "\n"
      << "near_field_tasks_for_common_edge="
      << profile.near_field_tasks_per_type[1].load() << "\n"
      << "near_field_tasks_for_common_vertex="
      << profile.near_field_tasks_per_type[2].load() << "\n"
      << "near_field_tasks_for_regular="
      << profile.near_field_tasks_per_type[3].load() << "\n";

  return out;
}

HBEM_NS_CLOSE

#endif // HIERBEM_INCLUDE_HMATRIX_HMATRIX_ASSEMBLY_PROFILE_H_
