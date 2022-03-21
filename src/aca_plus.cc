/**
 * @file aca_plus.cpp
 * @brief Introduction of aca_plus.cpp
 *
 * @date 2022-03-09
 * @author Jihuan Tian
 */

#include "aca_plus.h"

#include <random>

namespace IdeoBEM
{
  // Global definition of the random number device and generator.
  std::random_device rd;
#ifdef DEAL_II_WITH_64BIT_INDICES
  std::mt19937_64 rand_engine(rd());
#else
  std::mt19937 rand_engine(rd());
#endif

  size_type
  generate_random_index(const size_type a, const size_type b)
  {
    std::uniform_int_distribution<size_type> uniform_distribution(a, b);

    return uniform_distribution(rand_engine);
  }


  ACAConfig::ACAConfig()
    : max_iter(0)
    , epsilon(0.)
    , eta(0.)
  {}


  ACAConfig::ACAConfig(unsigned int v_max_iter, double v_epsilon, double v_eta)
    : max_iter(v_max_iter)
    , epsilon(v_epsilon)
    , eta(v_eta)
  {}
} // namespace IdeoBEM
