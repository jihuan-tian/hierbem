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
 * \file stl-memory-consumption-of-map.cc
 * \brief Verify the memory consumption calculation for @p std::map
 *
 * \ingroup test_cases stl
 * \author Jihuan Tian
 * \date 2022-05-06
 */

#include <iostream>
#include <map>

#include "utilities/generic_functors.h"

int
main()
{
  std::map<char, int> m{{'a', 10}, {'b', 20}, {'c', 30}, {'d', 40}, {'e', 100}};

  std::cout << "Memory consumption: " << memory_consumption_of_map(m)
            << std::endl;

  return 0;
}
