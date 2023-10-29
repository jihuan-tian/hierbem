/**
 * @file verify-deterministic-random-numbers.cc
 * @brief Verify generating a sequence of deterministic random numbers.
 *
 * @ingroup testers
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
