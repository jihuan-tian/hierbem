// Copyright (C) 2023-2025 Jihuan Tian <jihuan_tian@hotmail.com>
//
// This file is part of the HierBEM library.
//
// HierBEM is free software: you can use it, redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. The full text of the license can be found in the
// file LICENSE at the top level directory of HierBEM.

/**
 * @file verify-deterministic-random-numbers.cc
 * @brief Verify generating a sequence of deterministic random numbers.
 *
 * @ingroup test_cases
 * @author Jihuan Tian
 * @date 2023-10-29
 */

#include <iostream>
#include <random>

using namespace std;

int
main()
{
  std::mt19937                                rand_engine;
  std::uniform_int_distribution<unsigned int> uniform_distribution(1, 100);
  const unsigned int                          n = 100;
  for (unsigned i = 0; i < n; i++)
    {
      std::cout << uniform_distribution(rand_engine) << std::endl;
    }

  return 0;
}
