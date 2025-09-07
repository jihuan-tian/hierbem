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
